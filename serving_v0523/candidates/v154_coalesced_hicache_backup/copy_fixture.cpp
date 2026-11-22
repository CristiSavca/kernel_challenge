#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using cudaStream_t = int;
using cudaError_t = int;
#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
using CudaMemcpyBatchPtr = const void*;
#else
using CudaMemcpyBatchPtr = void*;
#endif
constexpr int cudaSuccess = 0, cudaErrorNotSupported = 1, cudaErrorCallRequiresNewerDriver = 2;
constexpr int cudaErrorInvalidValue = 3, cudaMemcpyDeviceToHost = 4, cudaMemLocationTypeDevice = 5;
constexpr int cudaMemLocationTypeHost = 6, cudaMemcpySrcAccessOrderStream = 7;
constexpr int stream_id = 19, device_id = 2;
int driver, error, batch_calls, queries;
struct cudaMemcpyAttributes {
  int srcAccessOrder, flags;
  struct Location { int type, id; } srcLocHint, dstLocHint;
};
struct Transfer { void* dst; const void* src; size_t bytes; };
std::vector<Transfer> pending;
namespace tvm::ffi {
struct TensorView {
  void* ptr;
  int64_t pitch;
  int bytes;
  void* data_ptr() const { return ptr; }
  int64_t stride(int axis) const { assert(axis==0); return pitch; }
  int dtype() const { return bytes; }
};
}
namespace host {
template<class... Args> void RuntimeCheck(bool ok, Args&&...) { assert(ok); }
void RuntimeDeviceCheck(int code) { assert(code==0); }
int dtype_bytes(int dtype) { return dtype; }
template<class T> std::vector<T> irange(T size) {
  std::vector<T> v(size); std::iota(v.begin(),v.end(),0); return v;
}
}
int cudaDriverGetVersion(int* value) { ++queries; *value=driver; return 0; }
void* get_cuda_memcpy_batch_async() { static int symbol; return &symbol; }
int cudaGetLastError() { return 0; }
const char* cudaGetErrorString(int) { return ""; }
int cudaMemcpyAsync(void* dst,const void* src,size_t size,int direction,int stream) {
  assert(direction==cudaMemcpyDeviceToHost && stream==stream_id && size>0);
  pending.push_back({dst,src,size}); return 0;
}
int call_cuda_memcpy_batch_async(void*,CudaMemcpyBatchPtr* dst,CudaMemcpyBatchPtr* src,size_t* bytes,size_t count,
                                cudaMemcpyAttributes* attrs,size_t* idx,size_t nattrs,int stream) {
  assert(nattrs==1 && idx[0]==0 && attrs->srcAccessOrder==cudaMemcpySrcAccessOrderStream);
  assert(attrs->srcLocHint.type==cudaMemLocationTypeDevice && attrs->srcLocHint.id==device_id);
  assert(attrs->dstLocHint.type==cudaMemLocationTypeHost && stream==stream_id);
  ++batch_calls;
  if(error) return error;
  for(size_t i=0;i<count;++i) pending.push_back({const_cast<void*>(dst[i]),src[i],bytes[i]});
  return 0;
}
// INSERT STOCK AND CANDIDATE HERE

using Tensor = tvm::ffi::TensorView;
using Batch = decltype(&original::try_copy_page_first_pages_batch);
using Plain = decltype(&original::copy_page_first_pages_fallback);
struct Counts { size_t descriptors, bytes; int batch, queried; bool accepted; };

Counts run(Batch batch,Plain copy,const std::vector<Tensor>& src,const std::vector<Tensor>& dst,
           const std::vector<int64_t>& ids,int page,int pages) {
  pending.clear(); batch_calls=queries=0;
  bool accepted = batch(src,dst,ids.data(),pages,page,device_id,stream_id);
  if(!accepted) copy(src,dst,ids.data(),pages,page,stream_id);
  size_t total=0;
  for(const auto& op:pending) { std::memcpy(op.dst,op.src,op.bytes); total+=op.bytes; }
  return {pending.size(),total,batch_calls,queries,accepted};
}
size_t cases=0,bytes=0,old_desc=0,new_desc=0,small_queries=0;
void check(int pages,int page,int elements,int elem_bytes,int tensors,int pattern,int driver_version,int status) {
  const int max_pages=3*pages+5;
  const size_t row_bytes=size_t(elements)*elem_bytes;
  const size_t guard=29;
  std::vector<int64_t> ids(pages*page, -1); size_t runs=0;
  for(int n=0;n<pages;++n) {
    int physical=n+1;
    if(pattern==1) physical=2*n+1;
    if(pattern==2) physical=pages-n;
    if(pattern==3) physical=n+(n/3)*2+1;
    if(pattern==4) physical=1+n%3; // sequential fallback preserves repeated destinations
    for(int r=0;r<page;++r) ids[n*page+r]=physical*page+r;
    runs+=n==0 || ids[n*page]!=ids[(n-1)*page]+page;
  }
  std::vector<std::vector<unsigned char>> source(tensors), reference(tensors), observed(tensors), oracle(tensors);
  std::vector<Tensor> src,dst0,dst1;
  for(int t=0;t<tensors;++t) {
    source[t].resize(guard*2+pages*page*row_bytes);
    for(size_t i=0;i<source[t].size();++i) source[t][i]=(i*13+(i>>8)+67*t+17)%251;
    oracle[t].assign(guard*2+max_pages*page*row_bytes,0xfe);
    reference[t]=observed[t]=oracle[t];
    src.push_back({source[t].data()+guard,elements,elem_bytes});
    dst0.push_back({reference[t].data()+guard,elements,elem_bytes});
    dst1.push_back({observed[t].data()+guard,elements,elem_bytes});
    for(int r=0;r<pages*page;++r)
      std::memcpy(oracle[t].data()+guard+ids[r]*row_bytes,source[t].data()+guard+r*row_bytes,row_bytes);
  }
  driver=driver_version; error=status;
  auto a=run(original::try_copy_page_first_pages_batch,original::copy_page_first_pages_fallback,src,dst0,ids,page,pages);
  auto b=run(candidate::try_copy_page_first_pages_batch,candidate::copy_page_first_pages_fallback,src,dst1,ids,page,pages);
  assert(a.accepted==b.accepted && a.batch==b.batch && a.bytes==b.bytes);
  assert(a.bytes==size_t(tensors)*pages*page*row_bytes && reference==oracle && observed==oracle);
  assert(a.descriptors==size_t(tensors)*pages);
  assert(b.descriptors==(b.accepted ? size_t(tensors)*pages : size_t(tensors)*runs));
  if(row_bytes*page<128*1024) { small_queries+=a.queried; assert(b.queried==0); }
  ++cases; bytes+=a.bytes; old_desc+=a.descriptors; new_desc+=b.descriptors;
}
int main() {
  for(int p:{0,1,2,7,63,64,65,129})
    for(int page:{1,3,16})
      for(int layout=0;layout<5;++layout)
        for(int tensors:{1,2})
          for(int el:{1,2})
            check(p,page,19,el,tensors,layout,12090,0);
  for(int stride:{1280,2048,5888})
    for(int layout=0;layout<5;++layout) check(64,1,stride,1,2,layout,12090,0);
  for(int size:{131071,131072,131073})
    for(int version:{12050,12090})
      for(int result:{0,1,2,3})
        for(int layout:{0,1,3}) check(4,1,size,1,2,layout,version,result);
  std::cout << "{\"cases\":" << cases << ",\"compared_payload_bytes\":" << bytes
            << ",\"stock_descriptors\":" << old_desc << ",\"candidate_descriptors\":" << new_desc
            << ",\"eliminated_small_page_driver_queries\":" << small_queries << "}\n";
}
