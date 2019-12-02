#include <iostream>
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

/// Base type for a color index
struct _Color
{
  /// Value beyond end
  static constexpr int64_t size=3;
};

/// Base type for a spin index
struct _Spin
{
  /// Value beyond end
  static constexpr int64_t size=4;
};

/// Tensor component defined by base type S
///
/// Inherit from S to get size
template <typename S>
struct TensorCompIdx : S
{
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
};

/// Spin index
using SpinIdx=TensorCompIdx<_Spin>;

/// Color index
using ColorIdx=TensorCompIdx<_Color>;

/// Tensor
template <typename...TC>
class Tens
{
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
    /// Size of this component
    const int64_t thisSize=std::remove_reference_t<T>::size;
    
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
  static constexpr int64_t size=product({TC::size...});
  
  /// Storage
  double data[size];
  
public:
  
  /// Access to inner data with any order
  template <typename...Cp>
  double& operator()(Cp&&...comps)
  {
    return data[reorderedIndex(std::forward<Cp>(comps)...)];
  }
  
  /// Gives trivial access
  double& trivialAccess(int64_t iSpin,int64_t iCol)
  {
    return data[iCol+3*iSpin];
  }
};

int main()
{
  /// Spindolor
  Tens<SpinIdx,ColorIdx> tensor;
  
  /// Fill the spincolor with flattened index
  for(SpinIdx i(0);i<4;i++)
    for(ColorIdx j(0);j<3;j++)
      tensor(i,j)=j+3*i;
  
  // Read components to access
  cout<<"Please enter spin and color index to printout: ";
  
  /// Spin component
  SpinIdx spin;
  cin>>spin;
  
  /// Color component
  ColorIdx col;
  cin>>col;
  
  asm("#here accessing (spin,col)");
  
  /// Spin,color access
  double sc=tensor(spin,col);
  
  asm("#here accessing (col,spin)");
  
  /// Color,spin access
  double cs=tensor(col,spin);
  
  asm("#here accessing trivially");
  
  /// Color,spin access
  double t=tensor.trivialAccess(spin,col);
  
  asm("#here printing");
  
  cout<<"Test: "<<sc<<" "<<cs<<" "<<t<<" expected: "<<col+3*spin<<endl;
  
  return 0;
}
