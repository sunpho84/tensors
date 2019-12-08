#include <iostream>
#include <memory>
#include <tuple>
#include <utility>

#if __cplusplus > 201402L

using std::as_const;

#else

template <class T>
constexpr const T& as_const(T& t) noexcept
{
  return t;
}

#endif

/// Remove \c const qualifier from anything
///
/// \todo  Check that  the  "dangling  reference forbidding"  below,
/// currently not working, is not actually necessary
template <typename T>
constexpr T& as_mutable(const T& v) noexcept
{
  return const_cast<T&>(v);
}

/// Provides also a non-const version of the method \c NAME
///
/// See
/// https://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
/// A const method NAME must be already present Example
///
/// \code
// class ciccio
/// {
///   double e{0};
///
/// public:
///
///   const double& get() const
///   {
///     return e;
///   }
///
///   PROVIDE_ALSO_NON_CONST_METHOD(get);
/// };
/// \endcode
#define PROVIDE_ALSO_NON_CONST_METHOD(NAME)				\
  /*! Overload the \c NAME const method passing all args             */ \
  template <typename...Ts> /* Type of all arguments                  */	\
  decltype(auto) NAME(Ts&&...ts) /*!< Arguments                      */ \
  {									\
    return as_mutable(as_const(*this).NAME(std::forward<Ts>(ts)...)); \
  }

template <typename T>
struct Crtp
{
  const T& operator~() const
  {
    return *static_cast<const T*>(this);
  }
  
  T& operator~()
  {
    return *static_cast<T*>(this);
  }
};

/// Compute the product of the passed list of T values
template <typename T=int64_t>
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

enum RowCol{ROW,COL};

/// Tensor component defined by base type S
///
/// Inherit from S to get size
template <typename S,
	  RowCol RC=ROW,
	  int Which=0>
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
  
  auto transp() const
  {
    return TensorCompIdx<S,(RC==ROW)?COL:ROW,Which>{i};
  }
};

/// Space index
template <RowCol RC=ROW,
	  int Which=0>
using SpaceIdx=TensorCompIdx<_Space,RC,Which>;

/// Spin index
template <RowCol RC=ROW,
	  int Which=0>
using SpinIdx=TensorCompIdx<_Spin,RC,Which>;

/// Color index
template <RowCol RC=ROW,
	  int Which=0>
using ColorIdx=TensorCompIdx<_Color,RC,Which>;

template <typename T>
using RefOrVal=std::conditional_t<std::is_lvalue_reference<T>::value,T&,T>;

template <typename T,
	  typename...C>
struct Bind
{
  RefOrVal<T> ref;
  
  std::tuple<RefOrVal<C>...> vals;
  
  template <typename...Tail>
  decltype(auto) operator()(Tail&&...tail) const
  {
    return ref(std::get<C>(vals)...,std::forward<Tail>(tail)...);
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator());
  
  
  template <typename F,
	    typename...Tail>
  decltype(auto) call(F&& f,
		      Tail&&...tail)
  {
    return f(ref,std::get<C>(vals)...,std::forward<Tail>(tail)...);
  }
  
  template <typename S>
  decltype(auto) operator[](S&& s) const
  {
    return (*this)(std::forward<S>(s));
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
  
  Bind(T&& ref,
       C&&...vals)
    : ref(ref),vals{vals...}
  {
  }
};

template <typename T>
auto bind(T&& ref)
{
  return ref;
}

template <typename T,
	  typename...Tp>
auto bind(T&& ref,
	  Tp&&...comps)
{
  return Bind<T,Tp...>(std::forward<T>(ref),std::forward<Tp>(comps)...);
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
    const int64_t staticSize=labs(product({TC::Base::sizeAtCompileTime...}));
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
  
  template <typename T>
  decltype(auto) operator[](T&& t)
  {
    return (*this)(std::forward<T>(t));
  }
  
  /// Gives trivial access
  double& trivialAccess(int64_t iSpin,int64_t iSpace,int64_t iCol,int64_t vol)
  {
    return data[iCol+3*(iSpace+vol*iSpin)];
  }
};

#define TEST(NAME,...)							\
  double& NAME(Tens<SpinIdx<ROW>,SpaceIdx<ROW>,ColorIdx<ROW>>& tensor,SpinIdx<ROW> spin,ColorIdx<ROW> col,SpaceIdx<ROW> space) \
  {									\
    asm("#here " #NAME "  access");					\
    return __VA_ARGS__;							\
  }

int64_t vol;


TEST(seq_fun,bind(bind(tensor,col),spin)(space))

TEST(bra_fun,tensor[col][spin][space])

TEST(csv_fun,tensor(col,spin,space));

TEST(svc_fun,tensor(spin,space,col));

TEST(hyp_fun,tensor(col)(spin)(space));

TEST(triv_fun,tensor.trivialAccess(spin,space,col,vol));

int main()
{
  using std::cout;
  using std::cin;
  using std::endl;
  
  cout<<"Please enter volume: ";
  cin>>vol;
  
  /// Spindolor
  Tens<SpinIdx<ROW>,SpaceIdx<ROW>,ColorIdx<ROW>> tensor(SpaceIdx<ROW>{vol});
  
  /// Fill the spincolor with flattened index
  for(SpinIdx<ROW> s(0);s<4;s++)
    for(SpaceIdx<ROW> v(0);v<vol;v++)
      for(ColorIdx<ROW> c(0);c<3;c++)
	tensor(s,v,c)=c+3*(v+vol*s);
  
  // Read components to access
  cout<<"Please enter spin, space and color index to printout: ";
  
  /// Spin component
  SpinIdx<ROW> spin;
  cin>>spin;
  
  /// Space component
  SpaceIdx<ROW> space;
  cin>>space;
  
  /// Color component
  ColorIdx<ROW> col;
  cin>>col;
  
  double& hyp=hyp_fun(tensor,spin,col,space);
  
  /// Spin,space,color access
  double& svc=svc_fun(tensor,spin,col,space);
  
  /// Color,spin,space access
  double& csv=csv_fun(tensor,spin,col,space);
  
  /// Color,spin,space access
  double& seq=seq_fun(tensor,spin,col,space);
  
  /// Color,spin,space access with []
  double& bra=bra_fun(tensor,spin,col,space);
  
  /// Trivial spin access
  double& t=triv_fun(tensor,spin,col,space);
  
  using SU3=Tens<ColorIdx<ROW>,ColorIdx<COL>>;
  
  SU3 link1,link2,link3;
  
  for(ColorIdx<ROW> i1{0};i1<3;i1++)
    for(ColorIdx<COL> k2{0};k2<3;k2++)
      {
	link3(i1,k2)=0.0;
	for(ColorIdx<COL> i2(0);i2<3;i2++)
	  link3(i1,k2)+=link1(i1,i2)*link2(i2.transp(),k2);
      }
  
  cout<<"Test: "<<svc<<" "<<csv<<" "<<t<<" "<<seq<<" "<<hyp<<" "<<bra<<" expected: "<<col+3*(space+vol*spin)<<endl;
  
  return 0;
}
