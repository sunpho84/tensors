#include <iostream>
#include <tuple>

using namespace std;

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

/// Tensor component
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
  /// Calculate the index - no more component to parse
  int64_t idx(int64_t outer) ///< Value of all the outer components
  {
    return outer;
  }
  
  /// Calculate index iteratively
  template <typename T,
	    typename...Tp>
  int64_t idx(int64_t outer,      ///< Value of all the outer components
	      T&& thisComp,       ///< Currently parsed component
	      Tp&&...innerComps)  ///< Inner components
  {
    /// Size of this component
    const int64_t thisSize=std::remove_reference_t<T>::size;
    
    /// Value of the index when including this component
    const int64_t thisVal=thisComp+thisSize*outer;
    
    return idx(thisVal,innerComps...);
  }
  
public:
  
  /// Cool feature of c++17 (avoidable at the cost of some more lines of code)
  static constexpr int64_t size=(TC::size*...);
  
  /// Storage
  double data[size];
  
  template <typename...Cp>
  double& operator()(Cp&&...comps)
  {
    /// Put the arguments in atuple
    auto argsInATuple=std::make_tuple(std::forward<Cp>(comps)...);
    
    /// Build the index reordering the components
    const int64_t i=idx(0,std::get<TC>(argsInATuple)...);
    
    /// Access data
    return data[i];
  }
};

int main()
{
  /// Spindolor
  Tens<SpinIdx,ColorIdx> a;
  
  /// Fill the spincolor with flattened index
  for(SpinIdx i(0);i<4;i++)
    for(ColorIdx j(0);j<3;j++)
      a(i,j)=j+3*i;
  
  // Read components to access
  cout<<"Please enter spin and color index: ";
  
  /// Spin component
  SpinIdx spin;
  cin>>spin;
  
  /// Color component
  ColorIdx col;
  cin>>col;
  
  asm("#here accessing (spin,col)");
  
  /// Spin,color access
  double sc=a(spin,col);
  
  asm("#here accessing (col,spin)");
  
  /// Color,spin access
  double cs=a(col,spin);
  
  asm("#here printing");
  
  cout<<"Test: "<<sc<<" "<<cs<<" expected: "<<col+3*spin<<endl;
  
  return 0;
}
