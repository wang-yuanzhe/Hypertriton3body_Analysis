CFLAGS=`root-config --cflags`
LDFLAGS=`root-config --ldflags --glibs` -lRooFit -lRooFitCore -lMinuit

#temp : temp.cpp RooATan.cxx RooATanDict.cxx libRooATan.so
#g++ -o temp temp.cpp RooATan.cxx RooATanDict.cxx $(CFLAGS) $(LDFLAGS)

libRooATan.so : libRooATan.so.1.0
	ln -sf libRooATan.so.1.0 libRooATan.so

libRooATan.so.1.0 : RooATan.o RooATanDict.o
	clang -shared -Wl,-install_name,libRooATan.dylib -o libRooATan.dylib RooATan.o RooATanDict.o
#gcc -dynamiclib,-Wl,-install_name,@rpath/libRooATan.so -o libRooATan.dylib RooATan.o RooATanDict.o

RooATanDict.o : RooATanDict.cxx
	g++ -c RooATanDict.cxx -fPIC $(CFLAGS) $(LDFLAGS)

RooATanDict.cxx : RooATan.h LinkDef.h
	rootcint -f RooATanDict.cxx -c RooATan.h LinkDef.h

RooATan.o : RooATan.cxx RooATan.h
	g++ -c RooATan.cxx -fPIC $(CFLAGS) $(LDFLAGS)
