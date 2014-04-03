################################################################################
#
# Build script for project
#
################################################################################
# $(BIN) est le nom du binaire généré
BIN = blank
# FLAG
FLAGS = -Wall -O3
# INCLUDES
INC = -I./include -I/usr/local/cuda/include
# Librairies
LIBDIR =
# Librairies
LIBS = -lGL -lGLU -ljpeg -lcuda -lcudart -lglut
# Librairies
OBJDIR = obj
# OBJET Directory
NVCC = nvcc
GCC = g++-4.3
GPP = g++-4.3
# $(OBJECTS) sont les objets des fichiers C / C++ standard qui seront générés après la compilation
OBJECTS = $(OBJDIR)/main.o
# $(OBJECTS) sont les objets des fichiers contenant du CUDA (.cu)
OBJECTS_CUDA = $(OBJDIR)/blank.cu.o

# all est la première règle à être exécutée car elle est la première
# dans le fichier Makefile. Notons que les dépendances peuvent être
# remplacées par une variable, ainsi que n'importe quel chaine de
# caractères des commandes
all: $(OBJECTS) $(OBJECTS_CUDA)
	@echo "**** PHASE DE LIEN ****"
	$(NVCC) --compiler-bindir=/usr/bin/gcc-4.3 $(LIBDIR) $(LIBS) $(OBJECTS) $(OBJECTS_CUDA) -o $(BIN)

$(OBJDIR)/%.o: %.cpp
	@echo "**** $@ ****"
	$(GPP) $(INC) $(FLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu
	@echo "**** $@ ****"
	$(NVCC) --compiler-bindir=/usr/bin/gcc-4.3 $(INC) -c $< -o $@

# Dependances particulieres
$(OBJDIR)/blank.cu.o : blankKernel.cu

clean:
	- rm $(OBJDIR)/*.o
	- rm $(BIN)


