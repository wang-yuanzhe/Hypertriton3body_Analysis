CXX = clang++
CFLAGS = `root-config --cflags`
LDFLAGS = `root-config --ldflags --glibs` -lRooFit -lRooFitCore -lMinuit

# 生成动态库（macOS使用.dylib）
libRooATan.dylib: RooATan.o RooATanDict.o
	$(CXX) -dynamiclib -o libRooATan.dylib RooATan.o RooATanDict.o -install_name @rpath/libRooATan.dylib $(LDFLAGS)

# 生成目标文件
RooATan.o: RooATan.cxx RooATan.h
	$(CXX) -c RooATan.cxx -fPIC $(CFLAGS)

RooATanDict.o: RooATanDict.cxx
	$(CXX) -c RooATanDict.cxx -fPIC $(CFLAGS)

RooATanDict.cxx: RooATan.h LinkDef.h
	rootcint -f RooATanDict.cxx -c RooATan.h LinkDef.h

