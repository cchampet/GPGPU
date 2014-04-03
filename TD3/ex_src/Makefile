################################################################################
#
# Build script for project
#
################################################################################
# $(BIN) est le nom du binaire généré
BIN = SAT
# FLAG
FLAGS = -Wall -g
# INCLUDES
INC = -I./include -I/usr/local/cuda/include
# Librairies
LIBDIR =
# Librairies
LIBS = -lGL -lGLU -lglut -lGLEW -ljpeg -lcuda -lcudart 
# Librairies
OBJDIR = obj
# OBJET Directory
NVCC = nvcc -arch=sm_11
GCC = g++-4.3
GPP = g++-4.3
# $(OBJECTS) sont les objets des fichiers C / C++ standard qui seront générés après la compilation
OBJECTS = $(OBJDIR)/main.o $(OBJDIR)/imageTexture.o
# $(OBJECTS) sont les objets des fichiers contenant du CUDA (.cu)
OBJECTS_CUDA = $(OBJDIR)/SAT.cu.o

# all est la première règle à être exécutée car elle est la première
# dans le fichier Makefile. Notons que les dépendances peuvent être
# remplacées par une variable, ainsi que n'importe quel chaine de
# caractères des commandes
all: $(OBJECTS) $(OBJECTS_CUDA)
	@echo "**** PHASE DE LIEN ****"
	$(NVCC) --compiler-bindir=/usr/bin/g++-4.3 $(LIBDIR) $(LIBS) $(OBJECTS) $(OBJECTS_CUDA) -o $(BIN)

$(OBJDIR)/%.o: %.cpp
	@echo "**** $@ ****"
	$(GPP) $(INC) $(FLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: %.cu
	@echo "**** $@ ****"
	$(NVCC) --compiler-bindir=/usr/bin/g++-4.3 $(INC) -c $< -o $@

# Dependances particulieres
$(OBJDIR)/SAT.cu.o : SATKernel.cu

clean:
	- rm $(OBJDIR)/*.o
	- rm $(BIN)


