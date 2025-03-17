// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME RooATanDict
#define R__NO_DEPRECATION

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "ROOT/RConfig.hxx"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Header files passed as explicit arguments
#include "RooATan.h"

// Header files passed via #pragma extra_include

// The generated code does not explicitly qualify STL entities
namespace std {} using namespace std;

namespace ROOT {
   static void *new_RooATan(void *p = nullptr);
   static void *newArray_RooATan(Long_t size, void *p);
   static void delete_RooATan(void *p);
   static void deleteArray_RooATan(void *p);
   static void destruct_RooATan(void *p);
   static void streamer_RooATan(TBuffer &buf, void *obj);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::RooATan*)
   {
      ::RooATan *ptr = nullptr;
      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< ::RooATan >(nullptr);
      static ::ROOT::TGenericClassInfo 
         instance("RooATan", ::RooATan::Class_Version(), "RooATan.h", 10,
                  typeid(::RooATan), ::ROOT::Internal::DefineBehavior(ptr, ptr),
                  &::RooATan::Dictionary, isa_proxy, 16,
                  sizeof(::RooATan) );
      instance.SetNew(&new_RooATan);
      instance.SetNewArray(&newArray_RooATan);
      instance.SetDelete(&delete_RooATan);
      instance.SetDeleteArray(&deleteArray_RooATan);
      instance.SetDestructor(&destruct_RooATan);
      instance.SetStreamerFunc(&streamer_RooATan);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::RooATan*)
   {
      return GenerateInitInstanceLocal(static_cast<::RooATan*>(nullptr));
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_DICT_(Init) = GenerateInitInstanceLocal(static_cast<const ::RooATan*>(nullptr)); R__UseDummy(_R__UNIQUE_DICT_(Init));
} // end of namespace ROOT

//______________________________________________________________________________
atomic_TClass_ptr RooATan::fgIsA(nullptr);  // static to hold class pointer

//______________________________________________________________________________
const char *RooATan::Class_Name()
{
   return "RooATan";
}

//______________________________________________________________________________
const char *RooATan::ImplFileName()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RooATan*)nullptr)->GetImplFileName();
}

//______________________________________________________________________________
int RooATan::ImplFileLine()
{
   return ::ROOT::GenerateInitInstanceLocal((const ::RooATan*)nullptr)->GetImplFileLine();
}

//______________________________________________________________________________
TClass *RooATan::Dictionary()
{
   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RooATan*)nullptr)->GetClass();
   return fgIsA;
}

//______________________________________________________________________________
TClass *RooATan::Class()
{
   if (!fgIsA.load()) { R__LOCKGUARD(gInterpreterMutex); fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::RooATan*)nullptr)->GetClass(); }
   return fgIsA;
}

//______________________________________________________________________________
void RooATan::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooATan.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      RooAbsPdf::Streamer(R__b);
      x.Streamer(R__b);
      a.Streamer(R__b);
      b.Streamer(R__b);
      R__b.CheckByteCount(R__s, R__c, RooATan::IsA());
   } else {
      R__c = R__b.WriteVersion(RooATan::IsA(), kTRUE);
      RooAbsPdf::Streamer(R__b);
      x.Streamer(R__b);
      a.Streamer(R__b);
      b.Streamer(R__b);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

namespace ROOT {
   // Wrappers around operator new
   static void *new_RooATan(void *p) {
      return  p ? new(p) ::RooATan : new ::RooATan;
   }
   static void *newArray_RooATan(Long_t nElements, void *p) {
      return p ? new(p) ::RooATan[nElements] : new ::RooATan[nElements];
   }
   // Wrapper around operator delete
   static void delete_RooATan(void *p) {
      delete (static_cast<::RooATan*>(p));
   }
   static void deleteArray_RooATan(void *p) {
      delete [] (static_cast<::RooATan*>(p));
   }
   static void destruct_RooATan(void *p) {
      typedef ::RooATan current_t;
      (static_cast<current_t*>(p))->~current_t();
   }
   // Wrapper around a custom streamer member function.
   static void streamer_RooATan(TBuffer &buf, void *obj) {
      ((::RooATan*)obj)->::RooATan::Streamer(buf);
   }
} // end of namespace ROOT for class ::RooATan

namespace {
  void TriggerDictionaryInitialization_RooATanDict_Impl() {
    static const char* headers[] = {
"RooATan.h",
nullptr
    };
    static const char* includePaths[] = {
"/opt/miniconda3/envs/alice/include/",
"/Users/yuanzhe/alice/H3L3Body/pdfLib/",
nullptr
    };
    static const char* fwdDeclCode = R"DICTFWDDCLS(
#line 1 "RooATanDict dictionary forward declarations' payload"
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_AutoLoading_Map;
class __attribute__((annotate("$clingAutoload$RooATan.h")))  RooATan;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(
#line 1 "RooATanDict dictionary payload"


#define _BACKWARD_BACKWARD_WARNING_H
// Inline headers
#include "RooATan.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[] = {
"RooATan", payloadCode, "@",
nullptr
};
    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("RooATanDict",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_RooATanDict_Impl, {}, classesHeaders, /*hasCxxModule*/false);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_RooATanDict_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_RooATanDict() {
  TriggerDictionaryInitialization_RooATanDict_Impl();
}
