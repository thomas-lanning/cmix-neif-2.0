"""Reduce weights of the least active early MLP channels on six discovery articles."""
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get('NEIF_SOURCE_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT/'training/tf/3_train/export/np2'))
import np2_container as np2

REFERENCE = '729694cba4969b7d833686ed3f597ff04d3b226da849d5f8004ec020a0ed9563'
CALIBRATION = '4813def8765538ba1fdf34423859ec85c189cd332ca5af643fd7853a25fcb3dc'
# Frozen on the original six C40 discovery windows, in estimated energy order.
RANKED = (
    (8, 808), (6, 1), (4, 797), (8, 638), (6, 436), (8, 70), (2, 325), (8, 632),
    (6, 76), (8, 830), (2, 55), (2, 47), (2, 200), (2, 789), (8, 1017), (8, 825),
    (4, 490), (4, 452), (2, 718), (8, 195), (8, 956), (6, 714), (2, 537), (2, 103),
    (8, 688), (2, 293), (4, 785), (4, 192), (8, 630), (8, 651), (8, 191), (6, 760),
    (8, 811), (2, 958), (8, 211), (6, 325), (6, 671), (2, 543), (4, 294), (8, 153),
    (8, 893), (6, 596), (6, 15), (4, 53), (2, 170), (8, 66), (8, 403), (6, 804),
    (6, 1014), (8, 669), (6, 89), (6, 301), (4, 1009), (8, 460), (6, 757), (2, 931),
    (4, 720), (2, 295), (4, 648), (2, 79), (6, 685), (6, 117), (8, 482), (8, 541),
    (2, 878), (2, 971), (2, 555), (6, 438), (4, 701), (2, 452), (8, 784), (2, 653),
    (4, 882), (2, 1019), (4, 666), (4, 689), (4, 690), (6, 669), (6, 856), (2, 717),
    (10, 65), (2, 631), (4, 488), (4, 260), (6, 570), (6, 690), (2, 709), (6, 860),
    (10, 495), (4, 236), (6, 850), (10, 124), (4, 826), (6, 171), (3, 129), (4, 424),
    (10, 636), (4, 985), (10, 618), (4, 609), (10, 313), (4, 560), (11, 510), (11, 542),
    (10, 1015), (3, 937), (4, 790), (10, 146), (10, 474), (10, 372), (10, 785), (10, 394),
    (3, 719), (11, 322), (10, 757), (10, 1023), (10, 429), (10, 718), (10, 900), (10, 840),
    (11, 539), (3, 583), (10, 639), (3, 143), (10, 784), (11, 250), (11, 891), (10, 399),
)


def float_scale(a):
    return (a.astype(np.uint32)<<16).view(np.float32).astype(np.float64)


def rank(tensors, candidates):
    entries = {t.name:t.a for t in tensors}
    scored = []
    for layer, ids, diagonal in candidates:
        name = f'blocks.{layer}.mlp.down.'
        q = entries[name+'weight.q'][:,ids].astype(np.float64)
        scale = float_scale(entries[name+'weight.scale']).reshape(-1,1)
        activation = float_scale(entries[name+'quantize_activation.scale']).item()
        predicted_energy = np.square(q*scale*activation).sum(0)*diagonal.astype(np.float64)
        scored.extend((loss,layer,int(unit)) for loss,unit in zip(predicted_energy,ids,strict=True))
    return tuple((layer,unit) for _,layer,unit in sorted(scored)[:128])


def reproduce_ranking(tensors, gram):
    if hashlib.sha256((gram/'manifest.json').read_bytes()).hexdigest()!=CALIBRATION:
        raise ValueError('discovery calibration changed')
    candidates=[]
    for layer in range(12):
        diag=np.load(gram/f'layer-{layer:02d}-down.npy',mmap_mode='r').diagonal()
        ids=np.argsort(diag,kind='stable')[:24]
        candidates.append((layer,ids,diag[ids].astype('<f2')))
    return rank(tensors,candidates)


def export(source, output, budget):
    if hashlib.sha256(source.read_bytes()).hexdigest()!=REFERENCE:
        raise ValueError('published weights changed')
    if budget!=int(budget) or abs(budget)>len(RANKED):raise ValueError('choose a signed channel count up to 128')
    tensors=np2.read_container(str(source))
    entries={t.name:t.a for t in tensors}
    for layer,unit in RANKED[:abs(int(budget))]:
        name=f'blocks.{layer}.mlp.'
        down=entries[name+'down.weight.q'][:,unit]
        up=entries[name+'up.weight.q'][unit]
        if budget<0:
            # Preserve endpoints and the stored per-output scales.
            down[:]=np.clip(2*np.rint(down.astype(float)/2),-7,7)
            up[:]=np.clip(2*np.rint(up.astype(float)/2),-7,7)
        else:
            down[:]=0
            # NP2 uses unique incoming rows as keys for its permutation code.
            # The outgoing column is zero, so an ID in these ten positions
            # has no effect on the active network. The encoder pays for it.
            up[:]=0
            up[:10]=[(unit>>j)&1 for j in range(10)]
            entries[name+'up.weight.scale'].reshape(-1)[unit]=0x3f80
    blob,_,permuted=np2.encode_np(tensors,True,np2.np_groups(tensors,True,True),True)
    with output.open('xb') as file:file.write(blob)
    actual=np2.read_container(str(output))
    if len(actual)!=len(permuted) or not all(a.same(b) for a,b in zip(actual,permuted,strict=True)):
        raise ValueError('candidate tensor round trip failed')
    return dict(channels=abs(int(budget)),mode='coarsen' if budget<0 else 'mute',packed_bytes=len(blob))


if __name__=='__main__':
    if sys.argv[1]=='rank':
        t=np2.read_container(str(ROOT/'weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5'))
        if reproduce_ranking(t,Path(sys.argv[2]))!=RANKED:raise ValueError('frozen ranking differs')
        print('Reproduced all 128 channel IDs')
    else:print(export(Path(sys.argv[1]),Path(sys.argv[2]),float(sys.argv[3])),flush=True)
