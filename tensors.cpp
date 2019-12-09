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

/// Returns true if T is a const lvalue reference
template <typename T>
constexpr bool is_const_lvalue_reference_v=std::is_lvalue_reference<T>::value and std::is_const<std::remove_reference_t<T>>::value;

/// Returns the type without "const" attribute if it is a reference
template <typename T>
decltype(auto) remove_const_if_ref(T&& t)
{
  return (std::conditional_t<is_const_lvalue_reference_v<T>,T&,T>)t;
}

template <typename T>
using ref_or_val_t=std::conditional_t<std::is_lvalue_reference<T>::value,T&,T>;

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
    return remove_const_if_ref(as_const(*this).NAME(std::forward<Ts>(ts)...)); \
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

/// Dynamic size
constexpr int64_t DYNAMIC=-1;

/// Base type for a space index
struct _Space
{
  /// Value beyond end
  static constexpr int64_t sizeAtCompileTime=DYNAMIC;
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

/// Row or column
enum RowCol{ROW,COL};

/// Tensor component defined by base type S
///
/// Inherit from S to get size
template <typename S,
	  RowCol RC=ROW,
	  int Which=0>
struct TensorCompIdx
{
  /// Base type
  typedef S Base;
  
  /// Value
  int64_t i;
  
  /// Check if the size is known at compile time
  static constexpr bool SizeKnownAtCompileTime=Base::sizeAtCompileTime!=DYNAMIC;
  
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
  
  /// Transposed index
  auto transp() const
  {
    return TensorCompIdx<S,(RC==ROW)?COL:ROW,Which>{i};
  }
};

/// Collection of components
template <typename...Tc>
using TensComps=std::tuple<Tc...>;

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

/// Binder a component or more than one
template <typename T,    // Type of the reference to bind
	  typename...C>  // Type of the components to bind
struct Binder
{
  /// Reference to bind
  ref_or_val_t<T> ref;
  
  /// Components to bind
  TensComps<ref_or_val_t<C>...> vals;
  
  /// Access to the reference passing all bound components, and more
  template <typename...Tail>
  decltype(auto) operator()(Tail&&...tail) const
  {
    return ref(std::get<C>(vals)...,std::forward<Tail>(tail)...);
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator());
  
  /// Call a function using the reference, the bound components, and all passed ones
  template <typename F,       // Type of the function
	    typename...Tail>  // Type of the other components
  decltype(auto) call(F&& f,                    ///< Function to call
		      Tail&&...tail) const      ///< Other components to pass
  {
    return f(ref,std::get<C>(vals)...,std::forward<Tail>(tail)...);
  }
  
  /// Single component access via subscribe operator
  template <typename S>                      // Type of the subscribed component
  decltype(auto) operator[](S&& s) const     ///< Subscribed component
  {
    return (*this)(std::forward<S>(s));
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
  
  /// Construct the binder from a reference and components
  Binder(T&& ref,     ///< Reference
	 C&&...vals)  ///< Components
    : ref(ref),vals{vals...}
  {
  }
};

// template <typename T>
// auto bind(T&& ref)
// {
//   return ref;
// }

/// Creates a binder, using the passed reference and component
template <typename T,    // Reference type
	  typename...Tp> // Components type
auto bind(T&& ref,       ///< Reference
	  Tp&&...comps)  ///< Components
{
  return Binder<T,Tp...>(std::forward<T>(ref),std::forward<Tp>(comps)...);
}

/// Tensor with Comps components, of Fund funamental type
template <typename Comps,
	  typename Fund>
class Tens;

/// Tensor
template <typename Fund,
	  typename...TC>
class Tens<TensComps<TC...>,Fund>
{
  /// Sizes of the components
  TensComps<TC...> sizes;
  
  /// Calculate the index - no more components to parse
  int64_t index(int64_t outer) const ///< Value of all the outer components
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
    const
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
  int64_t reorderedIndex(Cp&&...comps) const
  {
    /// Put the arguments in atuple
    auto argsInATuple=std::make_tuple(std::forward<Cp>(comps)...);
    
    /// Build the index reordering the components
    return index(0,std::get<TC>(argsInATuple)...);
  }
  
  /// Compute the data size
  int64_t size;
  
  /// Storage
  std::unique_ptr<Fund[]> data;
  
  /// Compute the size needed to initialize the tensor
  template <typename T>
  int64_t initSize(int64_t s)
  {
    /// Compile-time size
    constexpr int64_t cs=T::Base::sizeAtCompileTime;
    
    (int64_t&)(std::get<T>(sizes))=(cs>0)?cs:s;
    
    return std::get<T>(sizes);
  }
  
public:
  
  /// Initialize the tensor with the knowledge of the dynamic size
  template <typename...TD>
  Tens(TD&&...td)
  {
    /// Dynamic size
    const int64_t dynamicSize=product({initSize<TD>(std::forward<TD>(td))...});
    
    /// Static size
    const int64_t staticSize=labs(product({TC::Base::sizeAtCompileTime...}));
    
    size=dynamicSize*staticSize;
    //cout<<"Total size: "<<size<<endl;
    
    data=std::unique_ptr<Fund[]>(new Fund[size]);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)!=sizeof...(TC),void*> =nullptr>
  auto operator()(Cp&&...comps) const ///< Components
  {
    return bind(*this,comps...);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)==sizeof...(TC),void*> =nullptr>
  Fund& operator()(Cp&&...comps) const ///< Components
  {
    const int64_t i=reorderedIndex(std::forward<Cp>(comps)...);
    
    //cout<<"Index: "<<i<<endl;
    return data[i];
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator());
  
  /// Single component access via subscribe operator
  template <typename T>                   // Subscribed component type
  decltype(auto) operator[](T&& t) const  ///< Subscribed component
  {
    return (*this)(std::forward<T>(t));
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
  
  /// Provide trivial access to the fundamental data
  Fund& trivialAccess(const int64_t& i) const
  {
    return data[i];
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(trivialAccess);
};

/////////////////////////////////////////////////////////////////

using SpinSpaceColor=TensComps<SpinIdx<ROW>,SpaceIdx<ROW>,ColorIdx<ROW>>;

template <typename Fund>
using SpinColorField=Tens<SpinSpaceColor,Fund>;

using SpinColorFieldD=SpinColorField<double>;

#define TEST(NAME,...)							\
  double& NAME(SpinColorFieldD& tensor,SpinIdx<ROW> spin,ColorIdx<ROW> col,SpaceIdx<ROW> space) \
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

TEST(triv_fun,tensor.trivialAccess(col+3*(space+vol*spin)));

int main()
{
  using std::cout;
  using std::cin;
  using std::endl;
  
  cout<<"Please enter volume: ";
  cin>>vol;
  
  /// Spinspacecolor
  Tens<SpinSpaceColor,double> tensor(SpaceIdx<ROW>{vol});
  
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
  
  using SU3Comps=TensComps<ColorIdx<ROW>,ColorIdx<COL>>;
  
  using SU3=Tens<SU3Comps,double>;
  
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
