// Extract causal spelling continuations aligned to an authenticated mixer trace.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Event {
    uint32_t probability[4];
    uint16_t prefix;
    uint8_t previous, bit;
};
static_assert(sizeof(Event) == 20);

struct Votes { uint8_t ones = 0, count = 0; };
static_assert(sizeof(Votes) == 2);
struct Entry {
    uint64_t key = 0;
    std::array<uint8_t, 16> following{};
    unsigned count = 0, next = 0;
};

class Spelling {
    static constexpr uint64_t initial = 14695981039346656037ULL;
    std::vector<Entry> table = std::vector<Entry>(32768);
    uint64_t key = initial;
    unsigned letters = 0;
public:
    void reset() {
        std::fill(table.begin(), table.end(), Entry{});
        key = initial;
        letters = 0;
    }
    Votes predict(unsigned prefix) const {
        Votes votes;
        if (letters < 2 || letters > 32) return votes;
        const Entry& entry = table[key % table.size()];
        if (entry.key != key) return votes;
        const int shift = 7 - (31 - __builtin_clz(prefix));
        for (unsigned i = 0; i < entry.count; ++i) {
            unsigned next = (256u + entry.following[i]) >> shift;
            if ((next >> 1) == prefix) {
                ++votes.count;
                votes.ones += next & 1;
            }
        }
        return votes;
    }
    void observe(uint8_t value) {
        if (letters >= 2 && letters <= 32) {
            Entry& entry = table[key % table.size()];
            if (entry.key != key) {
                entry = {};
                entry.key = key;
            }
            entry.following[entry.next] = value;
            entry.next = (entry.next + 1) % entry.following.size();
            entry.count = std::min<unsigned>(entry.count + 1, entry.following.size());
        }
        if (value >= 'a' && value <= 'z') {
            key = (key ^ value) * 1099511628211ULL;
            ++letters;
        } else {
            letters = 0;
            key = initial;
        }
    }
};

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    const unsigned start = std::stoul(argv[2]), size = std::stoul(argv[3]);
    constexpr unsigned period = 1u << 20;
    const unsigned first = start - start % period;
    std::ifstream source(argv[1],std::ios::binary), trace(argv[4],std::ios::binary);
    if (!source || !trace || !size || size > 1000000 || start == 0) return 2;
    std::vector<uint8_t> bytes(start-first + size);
    source.seekg(first);
    if (!source.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) return 3;
    source.seekg(start-1);
    const unsigned previous = source.get();
    if (previous > 255) return 3;
    std::vector<Event> events;
    for (Event e; trace.read(reinterpret_cast<char*>(&e),sizeof e);) {
        if (e.prefix > 255 || e.bit > 1) return 4;
        events.push_back(e);
    }
    if (!trace.eof() || trace.gcount() || events.empty()) return 4;
    // Alignment is only for trace instrumentation. Prediction uses the byte prefix and
    // bytes already decoded. Forward and backward matches must establish the same offset.
    auto label = [&](int bit) {
        unsigned j = bit/8, at = start-first+j;
        unsigned prev = j ? bytes[at-1] : previous;
        return prev*512 + ((256u+bytes[at]) >> (7-bit%8));
    };
    auto event_label = [](const Event& e) { return unsigned(e.previous)*512 + 2*e.prefix + e.bit; };
    const int end = 8*int(size);
    std::vector<int> position(events.size(),-1);
    int cursor=0;
    for (size_t i=0; i<events.size(); ++i) {
        if (!events[i].prefix) continue;
        while (cursor<end && label(cursor)!=event_label(events[i])) ++cursor;
        if (cursor==end) throw std::runtime_error("trace does not match canonical stream");
        position[i]=cursor++;
    }
    cursor=end-1;
    for (size_t i=events.size(); i-- > 0;) {
        if (!events[i].prefix) continue;
        while (cursor>=0 && label(cursor)!=event_label(events[i])) --cursor;
        if (cursor!=position[i]) throw std::runtime_error("ambiguous trace alignment");
        --cursor;
    }
    Spelling spelling;
    unsigned next=first, last_byte=UINT32_MAX;
    std::ofstream out(argv[5], std::ios::binary);
    for (size_t i=0; i<events.size(); ++i) {
        Votes votes;
        if (position[i]>=0) {
            const unsigned current=start+position[i]/8;
            while (next<current) {
                if (next % period == 0) spelling.reset();
                spelling.observe(bytes[next-first]);
                ++next;
            }
            if (current!=last_byte && current%period==0) spelling.reset();
            last_byte=current;
            votes=spelling.predict(events[i].prefix);
        }
        out.write(reinterpret_cast<const char*>(&votes),sizeof votes);
    }
    out.close();
    return out ? 0 : 5;
}
