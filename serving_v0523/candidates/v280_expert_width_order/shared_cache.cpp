// Weighted LRU for actual intersections of A multicast boxes, packed B+scale tiles and output stores.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

struct Tile { int expert, row, begin, end, live; };
struct Fenwick {
  std::vector<int64_t> t;
  explicit Fenwick(int n):t(n+1) {}
  void add(int i,int v) { for (;i<int(t.size());i+=i&-i) t[i]+=v; }
  int64_t prefix(int i) const { int64_t v=0; for (;i;i-=i&-i)v+=t[i]; return v; }
};
int main(int argc,char** argv) {
  if(argc<7) return 2;
  int cohort=std::atoi(argv[1]),grid=std::atoi(argv[2]),fan=std::atoi(argv[3]),peers=std::atoi(argv[4]),allocate=std::atoi(argv[5]);
  const bool within_wave=cohort==0;
  if(within_wave)cohort=1;
  std::vector<Tile> tiles;
  int total=0;
  for(int i=6;i<argc;i++)total+=std::atoi(argv[i]);
  std::vector<int> cuts={0,total};
  int offset=0;
  for(int begin=6;begin<argc;begin+=cohort) {
    int m=tiles.size();
    for(int arg=begin;arg<std::min(argc,begin+cohort);arg++) {
      int n=std::atoi(argv[arg]);
      for(int off=0;off<n;off+=256) {
        int row=offset+off;
        cuts.push_back(row); cuts.push_back(std::min(row+256,total));
        tiles.push_back({arg-6,row,0,0,std::min(256,n-off)});
      }
      offset+=n;
    }
    auto width=[](int n){return n<=32?32:n<=128?128:n<=224?224:256;};
    bool desc=!(((begin-6)/cohort)%2);
    if(cohort>1)std::stable_sort(tiles.begin()+m,tiles.end(),[&](auto a,auto b) {return desc?width(a.live)>width(b.live):width(a.live)<width(b.live);});
  }
  if(within_wave) {
    int count=grid/32;
    if(count<=0 || grid%32)return 4;
    std::vector<int> prior(count,0);
    for(int begin=0;begin<int(tiles.size());begin+=count) {
      auto stop=std::min(int(tiles.size()),begin+count);
      std::vector<Tile> wave(tiles.begin()+begin,tiles.begin()+stop);
      auto issued=[](int n) {return n<=32?32:n<=128?128:n<=224?224:256;};
      std::stable_sort(wave.begin(),wave.end(),[&](auto a,auto b){return issued(a.live)>issued(b.live);});
      std::vector<int> lanes(wave.size());
      for(int i=0;i<int(lanes.size());++i)lanes[i]=i;
      std::stable_sort(lanes.begin(),lanes.end(),[&](int a,int b){return prior[a]<prior[b];});
      for(int i=0;i<int(lanes.size());++i) {
        tiles[begin+lanes[i]]=wave[i];
        prior[lanes[i]]+=issued(wave[i].live);
      }
    }
  }
  std::sort(cuts.begin(),cuts.end()); cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
  for(auto &t:tiles) {
    t.begin=std::lower_bound(cuts.begin(),cuts.end(),t.row)-cuts.begin();
    t.end=std::lower_bound(cuts.begin(),cuts.end(),std::min(t.row+256,total))-cuts.begin();
  }
  int n=tiles.size()*32;
  const int abase=(argc-6)*32*64,cbase=abase+cuts.size()*64;
  std::vector<int> last(cbase+n,0);
  int adds=n*(64+allocate);
  for(auto x:tiles) adds+=(x.end-x.begin)*64*(32/fan);
  Fenwick ft(adds);
  uint64_t resident=0;
  int now=0;
  std::array<int,5> capacities={18,27,36,40,50};
  std::array<std::array<uint64_t,5>,3> misses{};
  auto access=[&](int type,int key,int size) {
    int prev=last[key];
    uint64_t required=prev?resident-ft.prefix(prev)+size:UINT64_MAX;
    for(size_t i=0;i<capacities.size();++i)
      if(required>uint64_t(capacities[i])*1024*1024)misses[type][i]+=size;
    if(prev)ft.add(prev,-size);else resident+=size;
    ft.add(++now,size);last[key]=now;
  };
  for(int base=0;base<n;base+=grid) {
    int stop=std::min(n,base+grid);
    for(int first=base;first<stop;first+=peers) {
      int end=std::min(stop,first+peers);
      for(int k=0;k<64;k++)
        for(int b=first;b<end;b++) {
          auto t=tiles[b/32];
          if(b%fan==0)
            for(int s=t.begin;s<t.end;s++)access(0,abase+s*64+k,(cuts[s+1]-cuts[s])*128);
          access(1,(t.expert*32+b%32)*64+k,4608);
        }
      if(allocate)
        for(int b=first;b<end;b++) access(2,cbase+b,tiles[b/32].live*256);
    }
  }
  if(now!=adds)return 3;
  for(int i=0;i<5;i++)std::cout<<misses[0][i]/1048576.<<" "<<misses[1][i]/1048576.<<" ";
  std::cout<<"\n";
}
