#ifndef ROOATAN_H
#define ROOATAN_H

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"

class RooATan : public RooAbsPdf {
public:
  RooATan() {} ;
  RooATan(const char *name, const char *title,
	      RooAbsReal& _x,
	      RooAbsReal& _a,
	      RooAbsReal& _b);
  RooATan(const RooATan& other, const char* name=0) ;
  virtual TObject* clone(const char* newname) const { return new RooATan(*this,newname); }
  inline virtual ~RooATan() { }

  virtual Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* r=0) const;
  virtual Double_t analyticalIntegral(Int_t code,const char* rangeName=0) const;

protected:

  RooRealProxy x;
  RooRealProxy a;
  RooRealProxy b;

  Double_t evaluate() const ;

private:

  ClassDef(RooATan,1)
  // atan(a*x + b)
};

#endif
