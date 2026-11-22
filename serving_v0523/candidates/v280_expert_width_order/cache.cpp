// Fully associative packed-W2 sector LRU sensitivity; run on a host CPU.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

struct Tile { int expert, offset, width; };
struct Fenwick {
  std::vector<int> tree;
  explicit Fenwick(int n): tree(n + 1, 0) {}
  void add(int i, int d) { for (; i < int(tree.size()); i += i & -i) tree[i] += d; }
  int prefix(int i) const { int v = 0; for (; i; i -= i & -i) v += tree[i]; return v; }
};
int main(int argc,char** argv) {
  if(argc < 6) return 2;
  int cohort=std::atoi(argv[1]), grid=std::atoi(argv[2]), simultaneous=std::atoi(argv[3]);
  std::vector<Tile> tiles;
  for(int start=4;start<argc;start+=cohort) {
    auto from=tiles.size();
    for(int arg=start;arg<std::min(argc,start+cohort);++arg) {
      int rows=std::atoi(argv[arg]);
      for(int off=0;off<rows;off+=256) {
        int live=std::min(rows-off,256);
        tiles.push_back({arg-4,off,live<=32?32:live<=128?128:live<=224?224:256});
      }
    }
    if(cohort>1) {
      bool descending=cohort>=32 || !(((start-4)/cohort)%2);
      std::stable_sort(tiles.begin()+from,tiles.end(),[=](auto lhs,auto rhs) {
        return descending ? lhs.width>rhs.width : lhs.width<rhs.width;
      });
    }
  }
  int n=tiles.size()*32;
  std::vector<int> last((argc-4)*32*64,0), maxrows(grid);
  Fenwick recency(n*64);
  // Capacity is in 4,608-byte packed weight + scale sectors (N128, K64).
  std::vector<int> mib={1,4,9,12,18,24,27,32,36,40,50,64,90,144,288};
  std::vector<int> misses(mib.size());
  int now=0, distinct=0;
  for(int base=0;base<n;base+=grid) {
    int stop=std::min(n,base+grid);
    for(int cohortbase=base;cohortbase<stop;cohortbase+=simultaneous)
      for(int k=0;k<64;++k)
        for(int b=cohortbase;b<std::min(stop,cohortbase+simultaneous);++b) {
          const auto& tile=tiles[b/32];
          if(!k) maxrows[b-base]+=tile.width;
          int key=(tile.expert*32+b%32)*64+k;
          int old=last[key];
          int distance=old ? distinct-recency.prefix(old) : 100000000;
          if(old) recency.add(old,-1); else ++distinct;
          recency.add(++now,1);
          last[key]=now;
          for(unsigned j=0;j<mib.size();++j)
            if(distance >= mib[j]*1024*1024/4608) ++misses[j];
        }
  }
  std::cout << *std::max_element(maxrows.begin(),maxrows.end());
  for(int x:misses) std::cout << " " << x*4608.0/(1024*1024);
  std::cout << "\n";
}
