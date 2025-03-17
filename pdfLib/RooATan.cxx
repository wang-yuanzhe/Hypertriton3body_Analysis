#include "Riostream.h"

#include "RooATan.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include <math.h>
#include "TMath.h"

ClassImp(RooATan)

RooATan::RooATan(const char *name, const char *title,
  RooAbsReal& _x,
  RooAbsReal& _a,
  RooAbsReal& _b) :
  RooAbsPdf(name,title),
  x("x","x",this,_x),
  a("a","a",this,_a),
  b("b","b",this,_b)
  {
  }


  RooATan::RooATan(const RooATan& other, const char* name) :
  RooAbsPdf(other,name),
  x("x",this,other.x),
  a("a",this,other.a),
  b("b",this,other.b)
  {
  }



  Double_t RooATan::evaluate() const
  {
    return TMath::ATan(a * x + b) + TMath::Pi() / 2.;
  }


  Int_t RooATan::getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char*) const
  {
    if (matchArgs(allVars,analVars,x)) return 1 ;
    return 0 ;
  }

  Double_t RooATan::analyticalIntegral(Int_t code, const char* r) const
  {
    R__ASSERT(code==1);

    double xMin = a * x.min(r) + b;
    double xMax = a * x.max(r) + b;
    return ((xMax * TMath::ATan(xMax) - 0.5 * TMath::Log(1 + xMax * xMax)) -
           (xMin * TMath::ATan(xMin) - 0.5 * TMath::Log(1 + xMin * xMin))) / a + TMath::Pi() / 2. * (x.max(r) - x.min(r));
  }
