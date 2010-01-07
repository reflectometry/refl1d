include /Users/pkienzle/danse/reflectometry/trunk/garefl/examples/xray/../.././Makeconf

# Files produced by the fit programs
LOGFILES=profile*.dat fit*.dat* fit.log par.dat pop.dat gmon.out

all: fit$(EXE) gaplot

fit$(EXE): setup.o $(GAREFLDEP) $(MODEL1DDEP) $(FITDEP)
	$(CXX) $(LDFLAGS) -o $@ $< $(GAREFLLIB) $(MODEL1DLIB) $(FITLIB) $(FLIBS) $(LIBS)

gaplot:
	$(LN_S) $(GAREFL_INCDIR)/gaplot gaplot

$(GAREFLDEP) $(MODEL1DDEP) $(FITDEP): builddeps ;

builddeps:
	cd $(FIT_LIBDIR) && $(MAKE)
	cd $(MODEL1D_LIBDIR) && $(MAKE)
	cd $(GAREFL_LIBDIR) && $(MAKE)

#setup.o: setup.c $(FITDEPS)

#setup$(DLL): setup.o
#	$(CXX) -shared -o $@ $< $(MODEL1DLIB) $(FLIBS) $(LIBS)

clean: ; -$(RM) setup.o

distclean: clean ; -$(RM) *~ fit$(EXE) $(LOGFILES) Makefile gaplot

check: fit
	./fit -n 2
