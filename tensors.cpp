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

/// Filter a tuple on the basis of a predicate on the type
///
/// Internal implementation working out a single type, forward
/// declaration
template <bool,
	  typename>
struct _TupleFilter;

/// Filter a tuple on the basis of a predicate
///
/// True case, in which the type is filtered
template <typename T>
struct _TupleFilter<true,T>
{
  /// Helper type, used to cat the results
  using type=std::tuple<T>;
  
  /// Filtered value
  const type value;
  
  /// Construct, taking a tuple type and filtering the valid casis
  template <typename Tp>
  _TupleFilter(Tp&& t) : ///< Tuple to filter
    value{std::get<T>(t)}
  {
  }
};

/// Filter a tuple on the basis of a predicate
///
/// True case, in which the type is filtered out
template <typename T>
struct _TupleFilter<false,T>
{
  /// Helper empty type, used to cat the results
  using type=std::tuple<>;
  
  /// Empty value
  const type value;
  
  /// Construct without getting the type
  template <typename Tp>
  _TupleFilter(Tp&& t) ///< Tuple to filter
  {
  }
};

/// Returns a tuple, filtering out the non needed types
template <template <class> class F,          // Predicate to be applied on the types
	  typename...T>                      // Types contained in the tuple to be filtered
auto tupleFilter(const std::tuple<T...>& t) ///< Tuple to filter
{
  return std::tuple_cat(_TupleFilter<F<T>::value,T>{t}.value...);
}

/// Type obtained applying the predicate filter F on the tuple T
template <template <class> class F,
	  typename T>
using TupleFilter=decltype(tupleFilter<F>(T{}));

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

// /// Compute the product of the passed list of T values
// template <typename T,
// 	  typename...Ts>
// constexpr T prodsuct(Ts&&...list)
// {
//   /// Result
//   T out=1;
  
//   const T l[]{list...};
  
//   for(auto i : l)
//     out*=i;
  
//   return out;
// }

/// Compute the product of the passed list of T values
template <typename F,
	  typename T,
	  typename...Ts>
constexpr T combine(F&& f,
		    const T& init,
		    Ts&&...list)
{
  /// Result
  T out=init;

  const T l[]{list...};
  
  for(auto i : l)
    out=f(out,i);
  
  return out;
}

template <typename T,
	  typename...Ts>
constexpr auto product(Ts&&...t)
{
  return combine(std::multiplies<>(),T{1},std::forward<Ts>(t)...);
}

///Type used to specify size
using Size=int64_t;

/// Dynamic size
constexpr Size DYNAMIC=-1;

/// Specify the size at compile time
template <Size SIZE=DYNAMIC>
struct TensCompSize
{
  /// Value beyond end
  static constexpr Size sizeAtCompileTime()
  {
    return SIZE;
  };
};

/// Base type for a space index
struct _Space : public TensCompSize<>
{
};

/// Base type for a color index
struct _Color : public TensCompSize<3>
{
};

/// Base type for a spin index
struct _Spin : public TensCompSize<4>
{
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
  Size i;
  
  /// Check if the size is known at compile time
  static constexpr bool SizeIsKnownAtCompileTime=Base::sizeAtCompileTime()!=DYNAMIC;
  
  /// Init from int
  explicit TensorCompIdx(Size i) : i(i)
  {
  }
  
  /// Default constructor
  TensorCompIdx()
  {
  }
  
  /// Convert to actual value
  operator Size&()
  {
    return i;
  }
  
  /// Convert to actual value
  operator const Size&() const
  {
    return i;
  }
  
  /// Transposed index
  auto transp() const
  {
    return TensorCompIdx<S,(RC==ROW)?COL:ROW,Which>{i};
  }
};

/// Predicate returning whether the size is known ow not at compile time
template <bool Asked=true>
struct SizeIsKnownAtCompileTime
{
  /// Internal implementation
  template <typename T>
  struct t
  {
    /// Predicate result
    static constexpr bool value=(T::SizeIsKnownAtCompileTime==Asked);
  };
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

template <typename Fund,
	  Size StaticSize,
	  bool AllStatic>
struct TensStorage
{
  struct DynamicStorage
  {
    /// Storage
    std::unique_ptr<Fund[]> data;
    
    DynamicStorage(const Size& dynSize) : data(new Fund[StaticSize*dynSize])
    {
    }
  };
  
  struct StackStorage
  {
    /// Storage
    Fund data[StaticSize];
    
    StackStorage(const Size&)
    {
    }
  };
  
  static constexpr Size MAX_STACK_SIZE=2304;
  
  static constexpr bool stackAllocated=AllStatic and StaticSize*sizeof(Fund)<=MAX_STACK_SIZE;
  
  std::conditional_t<stackAllocated,StackStorage,DynamicStorage> data;
  
  TensStorage(const Size& size) : data(size)
  {
  }
  
  /// Single component access via subscribe operator
  template <typename T>                   // Subscribed component type
  auto& operator[](const T& t) const  ///< Subscribed component
  {
    return data.data[t];
  }
  
  PROVIDE_ALSO_NON_CONST_METHOD(operator[]);
  
};

/// Tensor with Comps components, of Fund funamental type
template <typename Comps,
	  typename Fund>
class Tens;

/// Tensor
template <typename Fund,
	  typename...TC>
class Tens<TensComps<TC...>,Fund>
{
  using StaticComps=TupleFilter<SizeIsKnownAtCompileTime<true>::t,TensComps<TC...>>;
  
  using DynamicComps=TupleFilter<SizeIsKnownAtCompileTime<false>::t,TensComps<TC...>>;
  
  /// Sizes of the dynamic components
  const DynamicComps dynamicSizes;
  
  /// Static size
  static constexpr Size staticSize=product<Size>((TC::SizeIsKnownAtCompileTime?TC::Base::sizeAtCompileTime():1)...);
  
  /// Calculate the index - no more components to parse
  Size index(Size outer) const ///< Value of all the outer components
  {
    return outer;
  }
  
  template <typename Tv,
	    std::enable_if_t<Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
  constexpr Size compSize() const
  {
    return Tv::Base::sizeAtCompileTime();
  }
  
  template <typename Tv,
	    std::enable_if_t<not Tv::SizeIsKnownAtCompileTime,void*> =nullptr>
  const Size& compSize() const
  {
    return std::get<Tv>(dynamicSizes);
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
  Size index(Size outer,      ///< Value of all the outer components
	     T&& thisComp,       ///< Currently parsed component
	     Tp&&...innerComps)  ///< Inner components
    const
  {
    /// Remove reference to access to types
    using Tv=std::remove_reference_t<T>;
    
    /// Size of this component
    const Size thisSize=compSize<Tv>();
    
    //cout<<"thisSize: "<<thisSize<<endl;
    /// Value of the index when including this component
    const Size thisVal=outer*thisSize+thisComp;
    
    return index(thisVal,innerComps...);
  }
  
  /// Intermediate layer to reorder the passed components
  template <typename...Cp>
  Size reorderedIndex(Cp&&...comps) const
  {
    /// Put the arguments in atuple
    auto argsInATuple=std::make_tuple(std::forward<Cp>(comps)...);
    
    /// Build the index reordering the components
    return index(0,std::get<TC>(argsInATuple)...);
  }
  
  /// Compute the data size
  Size size;
  
  static constexpr bool allCompsAreStatic=std::is_same<DynamicComps,std::tuple<>>::value;
  
  /// Storage
  TensStorage<Fund,staticSize,allCompsAreStatic> data;
  
  template <typename Ds,
	    typename Out>
  Size initializeDynSize(const Ds& inputs,
			 Out& out)
  {
    out=std::get<Out>(inputs);
    
    return out;
  }
  
  /// Compute the size needed to initialize the tensor and set it
  template <typename...Td,
	    typename...T>
  TensComps<Td...> initializeDynSizes(TensComps<Td...>*,
				      T&&...in)
  {
    static_assert(sizeof...(T)==sizeof...(Td),"Number of passed dynamic sizes not matching the needed one");
    
    return {std::get<Td>(std::make_tuple(in...))...};
  }
  
public:
  
  static constexpr bool stackAllocated=decltype(data)::stackAllocated;
  
  /// Initialize the tensor with the knowledge of the dynamic size
  template <typename...TD>
  Tens(TD&&...td) :
    dynamicSizes{initializeDynSizes((DynamicComps*)nullptr,std::forward<TD>(td)...)},
    data(product<Size>(std::forward<TD>(td)...))
  {
    /// Dynamic size
    //const Size dynamicSize=product<Size>(std::forward<TD>(td)...);
    
    //size=dynamicSize*staticSize;
    //cout<<"Total size: "<<size<<endl;
    
    //data=std::unique_ptr<Fund[]>(new Fund[size]);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)!=sizeof...(TC),void*> =nullptr>
  decltype(auto) operator()(Cp&&...comps) const ///< Components
  {
    return bind(*this,comps...);
  }
  
  /// Access to inner data with any order
  template <typename...Cp,
	    std::enable_if_t<sizeof...(Cp)==sizeof...(TC),void*> =nullptr>
  decltype(auto) operator()(Cp&&...comps) const ///< Components
  {
    const Size i=reorderedIndex(std::forward<Cp>(comps)...);
    
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
  Fund& trivialAccess(const Size& i) const
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

Size vol;

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
  
  // for(ColorIdx<ROW> i1{0};i1<3;i1++)
  //   for(ColorIdx<COL> k2{0};k2<3;k2++)
  //     {
  // 	link3(i1,k2)=0.0;
  // 	for(ColorIdx<COL> i2(0);i2<3;i2++)
  // 	  link3(i1,k2)+=link1(i1,i2)*link2(i2.transp(),k2);
  //     }
  
  cout<<"Test: "<<svc<<" "<<csv<<" "<<t<<" "<<seq<<" "<<hyp<<" "<<bra<<" expected: "<<col+3*(space+vol*spin)<<endl;
  
  cout<<SpinColorFieldD::stackAllocated<<endl;
  cout<<SU3::stackAllocated<<endl;
  
  return 0;
}
