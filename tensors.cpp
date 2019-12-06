#include <iostream>
#include <memory>
#include <tuple>

using namespace std;

/// Compute the product of the passed list of T values
template <typename T>
constexpr T product(const std::initializer_list<T>& list)
{
  /// Result
  T out=1;
  
  for(auto i : list)
    out*=i;
  return out;
}

/// Base type for a space index
struct _Space
{
  /// Value beyond end
  static constexpr int64_t sizeAtCompileTime=-1;
};

/// Base type for a color index
struct _Color
{
  /// Value beyond end
  static constexpr int64_t sizeAtCompileTime=3;
};

/// Base type for a spin index
struct _Spin
{
  /// Value beyond end
  static constexpr int64_t sizeAtCompileTime=4;
};

/// Tensor component defined by base type S
///
/// Inherit from S to get size
template <typename S>
struct TensorCompIdx
{
  typedef S Base;
  
  /// Value
  int64_t i;
  
  /// Init from int
  explicit TensorCompIdx(int64_t i) : i(i)
  {
  }
  
  /// Default constructor
  TensorCompIdx()
  {
  }
  
  /// Convert to actual value
  operator int64_t&()
  {
    return i;
  }
  
  /// Convert to actual value
  operator const int64_t&() const
  {
    return i;
  }
};

/// Space index
using SpaceIdx=TensorCompIdx<_Space>;

/// Spin index
using SpinIdx=TensorCompIdx<_Spin>;

/// Color index
using ColorIdx=TensorCompIdx<_Color>;

template <typename T,
	  typename C>
struct Bind
{
  std::conditional_t<std::is_lvalue_reference<T>::value,T&,T> ref;
  
  C val;
  
  template <typename...Tail>
  decltype(auto) operator()(Tail&&...tail)
  {
    return ref(val,std::forward<Tail>(tail)...);
  }
  
  Bind(T&& ref,
       C&& val)
    : ref(ref),val(val)
  {
  }
};

template <typename T>
auto bind(T&& ref)
{
  return ref;
}

template <typename T,
	  typename Head,
	  typename...Tail>
auto bind(T&& ref,
	  Head&& head,
	  Tail&&...tail)
{
  return bind(Bind<T,Head>(std::forward<T>(ref),std::forward<Head>(head)),std::forward<Tail>(tail)...);
}

/// Tensor
template <typename...TC>
class Tens
{
  std::tuple<TC...> sizes;
  
  /// Calculate the index - no more components to parse
  int64_t index(int64_t outer) ///< Value of all the outer components
  {
    return outer;
  }
  
  /// Calculate index iteratively
  ///
  /// Given the components (i,j,k) we must compute ((0*ni+i)*nj+j)*nk+k
  ///
  /// The parsing of the variadic components is done left to right, so
  /// to compute the nested bracket list we must proceed inward. Let's
  /// say we are at component j. We define outer=(0*ni+i) the result
  /// of inner step. We need to compute thisVal=outer*nj+j and pass it
  /// to the inner step, which incorporate iteratively all the inner
  /// components. The first step requires outer=0.
  template <typename T,
	    typename...Tp>
  int64_t index(int64_t outer,      ///< Value of all the outer components
		T&& thisComp,       ///< Currently parsed component
		Tp&&...innerComps)  ///< Inner components
  {
    constexpr int64_t i=std::remove_reference_t<T>::Base::sizeAtCompileTime;
    
    /// Size of this component
    const int64_t thisSize=(i>0)?i:std::get<std::remove_reference_t<T>>(sizes);
    
    //cout<<"thisSize: "<<thisSize<<endl;
    /// Value of the index when including this component
    const int64_t thisVal=outer*thisSize+thisComp;
    
    return index(thisVal,innerComps...);
  }
  
  /// Intermediate layer to reorder the passed components
  template <typename...Cp>
  int64_t reorderedIndex(Cp&&...comps)
  {
    /// Put the arguments in atuple
    auto argsInATuple=std::make_tuple(std::forward<Cp>(comps)...);
    
    /// Build the index reordering the components
    return index(0,std::get<TC>(argsInATuple)...);
  }
  
  /// Compute the data size
  int64_t size;
  
  /// Storage
  std::unique_ptr<double[]> data;
  
  template <typename T>
  int64_t initSize(int64_t s)
  {
    constexpr int64_t i=T::Base::sizeAtCompileTime;
    (int64_t&)(std::get<T>(sizes))=(i>0)?i:s;
    return std::get<T>(sizes);
  }
  
public:
  
  //
  template <typename...TD>
  Tens(TD&&...td)
  {
    const int64_t dynamicSize=product({initSize<TD>(std::forward<TD>(td))...});
    const int64_t staticSize=abs(product({TC::Base::sizeAtCompileTime...}));
    size=dynamicSize*staticSize;
    //cout<<"Total size: "<<size<<endl;
    
    data=std::unique_ptr<double[]>(new double[size]);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)!=sizeof...(TC),void*> =nullptr>
  auto operator()(Cp&&...comps)
  {
    return bind(*this,comps...);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)==sizeof...(TC),void*> =nullptr>
  double& operator()(Cp&&...comps)
  {
    const int64_t i=reorderedIndex(std::forward<Cp>(comps)...);
    
    //cout<<"Index: "<<i<<endl;
    return data[i];
  }
  
  /// Gives trivial access
  double& trivialAccess(int64_t iSpin,int64_t iSpace,int64_t iCol,int64_t vol)
  {
    return data[iCol+3*(iSpace+vol*iSpin)];
  }
};

#define TEST(NAME,...)							\
  double& NAME(Tens<SpinIdx,SpaceIdx,ColorIdx>& tensor,SpinIdx spin,ColorIdx col,SpaceIdx space) \
  {									\
    asm("#here " #NAME "  access");					\
    return __VA_ARGS__;							\
  }

int64_t vol;


TEST(seq_fun,bind(bind(tensor,col),spin)(space))

TEST(csv_fun,tensor(col,spin,space));

TEST(svc_fun,tensor(spin,space,col));

TEST(hyp_fun,tensor(col)(spin)(space));

TEST(triv_fun,tensor.trivialAccess(spin,space,col,vol));

int main()
{
  cout<<"Please enter volume: ";
  cin>>vol;
  
  /// Spindolor
  Tens<SpinIdx,SpaceIdx,ColorIdx> tensor(SpaceIdx{vol});
  
  /// Fill the spincolor with flattened index
  for(SpinIdx s(0);s<4;s++)
    for(SpaceIdx v(0);v<vol;v++)
      for(ColorIdx c(0);c<3;c++)
	tensor(s,v,c)=c+3*(v+vol*s);
  
  // Read components to access
  cout<<"Please enter spin, space and color index to printout: ";
  
  /// Spin component
  SpinIdx spin;
  cin>>spin;
  
  /// Space component
  SpaceIdx space;
  cin>>space;
  
  /// Color component
  ColorIdx col;
  cin>>col;
  
  __asm volatile ("");
  double& hyp=hyp_fun(tensor,spin,col,space);
  
  asm("#here accessing (spin,space,col)");
  cin>>spin;
  cin>>space;
  cin>>col;
  __asm volatile ("");
    
  /// Spin,space,color access
  double& svc=svc_fun(tensor,spin,col,space);
  
  asm("#here accessing (col,spin,space)");
  cin>>spin;
  cin>>space;
  cin>>col;
  __asm volatile ("");
  /// Color,spin,space access
  double& csv=csv_fun(tensor,spin,col,space);
  
  asm("#here accessing sequentially");
  cin>>spin;
  cin>>space;
  cin>>col;
  __asm volatile ("");
  
  /// Color,spin,space access
  double& seq=seq_fun(tensor,spin,col,space);
  
  asm("#here accessing trivially");
  cin>>spin;
  cin>>space;
  cin>>col;
  __asm volatile ("");
  /// Trivial spin access
  double& t=triv_fun(tensor,spin,col,space);
  
  asm("#here printing");
  
  cout<<"Test: "<<svc<<" "<<csv<<" "<<t<<" "<<seq<<" "<<hyp<<" expected: "<<col+3*(space+vol*spin)<<endl;
  
  return 0;
}
