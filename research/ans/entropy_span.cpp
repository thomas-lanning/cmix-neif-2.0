// SPDX-License-Identifier: GPL-3.0-only
// Replay native coder intervals, including the carry state at the first bit.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

struct Event {
  uint32_t p, head;
  uint16_t prefix;
  uint8_t previous, bit;
};
static_assert(sizeof(Event) == 12);
constexpr uint64_t total = uint64_t{1} << 32;

struct Trace {
  uint32_t low, high, offset;
  std::vector<Event> events;
  explicit Trace(const char* path) {
    std::ifstream file(path, std::ios::binary);
    uint32_t header[3];
    if (!file.read(reinterpret_cast<char*>(header), sizeof header)) throw std::runtime_error("short header");
    low=header[0]; high=header[1]; offset=header[2];
    if (low>=high) throw std::runtime_error("invalid starting interval");
    Event event;
    while (file.read(reinterpret_cast<char*>(&event),sizeof event)) {
      if (!event.p || !event.head || event.bit>1) throw std::runtime_error("invalid event");
      events.push_back(event);
    }
    if (!file.eof() || file.gcount() || events.empty()) throw std::runtime_error("incomplete events");
  }
};

static unsigned byte_class(unsigned c) {
  if (c>='a' && c<='z') return 0;
  if (c>='A' && c<='Z') return 1;
  if (c>='0' && c<='9') return 2;
  if (c<=' ') return 3;
  if (c>=128) return 4;
  return 5;
}

static unsigned context(const Event& e,unsigned mode) {
  unsigned side=e.p>=total/2;
  uint32_t error=side ? total-e.p : e.p;
  unsigned confidence=std::min(7u,static_cast<unsigned>(__builtin_clz(error)));
  unsigned depth=(31u-__builtin_clz(static_cast<unsigned>(e.prefix)|1))&7;
  unsigned base=confidence*2+side;
  if (mode==1) return base;
  if (mode==2) return base*8+depth;
  if (mode==3) return base*6+byte_class(e.previous);
  if (mode==4) return base*3+(e.head<e.p ? 0 : e.head>e.p ? 2 : 1);
  if (mode==5) return base*4+(e.prefix&3);
  throw std::runtime_error("unknown family");
}

struct Cell { int32_t residual=0, variance=0; unsigned n=0; };

// Online Newton step for an additive correction to log-odds. Store sufficient
// statistics in Q12 and clip the step before its Q32 probability update.
static uint32_t corrected(const Event& e,Cell& c,unsigned prior,unsigned strength) {
  int64_t slope=(uint64_t(e.p)*(total-e.p))>>32;
  int64_t step=std::clamp<int64_t>((int64_t(c.residual)*256)/(int64_t(prior)*4096+c.variance),-128,128);
  int64_t p=int64_t(e.p)+(slope*step*strength)/1024;
  c.residual+=int32_t((int64_t(e.bit)*int64_t(total)-e.p)/1048576);
  c.variance+=int32_t((slope+524288)/1048576);
  if (++c.n==4096) { c.n/=2; c.residual/=2; c.variance/=2; }
  return static_cast<uint32_t>(std::clamp<int64_t>(p,1,total-1));
}

static size_t replay(const Trace& t,unsigned family,unsigned prior,unsigned strength,
                     const std::vector<uint8_t>& oracle) {
  std::array<Cell,128> cells{};
  uint32_t low=t.low,high=t.high;
  size_t bytes=0;
  for (const auto& e:t.events) {
    uint32_t p=e.p;
    if (family && e.prefix) p=corrected(e,cells.at(context(e,family)),prior,strength);
    uint32_t mid=low+static_cast<uint32_t>((uint64_t(high-low)*p)>>32);
    if (e.bit) high=mid; else low=mid+1;
    while ((low>>24)==(high>>24)) {
      if (!family && (t.offset+bytes>=oracle.size() || oracle[t.offset+bytes]!=(low>>24)))
        throw std::runtime_error("control differs from published byte "+std::to_string(t.offset+bytes));
      ++bytes; low<<=8; high=(high<<8)|255;
    }
  }
  return bytes;
}

int main(int argc,char** argv) try {
  if (argc!=3 && argc!=6) return 2;
  Trace trace(argv[1]);
  std::ifstream source(argv[2],std::ios::binary);
  if (!source) throw std::runtime_error("missing oracle");
  std::vector<uint8_t> oracle((std::istreambuf_iterator<char>(source)),{});
  const size_t base=replay(trace,0,1,1,oracle);
  auto measure=[&](unsigned family,unsigned prior,unsigned strength) {
    std::cout<<family<<' '<<prior<<' '<<strength<<' '<<trace.events.size()<<' '
             <<trace.offset<<' '<<base<<' '<<replay(trace,family,prior,strength,oracle)<<'\n';
  };
  if (argc==6) {
    unsigned family=std::stoul(argv[3]),prior=std::stoul(argv[4]),strength=std::stoul(argv[5]);
    if (family>5 || !prior || prior>1024 || !strength || strength>4) return 2;
    measure(family,prior,strength);
  } else {
    measure(0,1,1);
    for (unsigned family=1;family<=5;++family)
      for (unsigned prior:{8u,32u,128u,512u})
        for (unsigned strength:{1u,2u,4u}) measure(family,prior,strength);
  }
  return 0;
} catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
