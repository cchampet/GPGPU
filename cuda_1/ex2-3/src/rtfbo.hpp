#ifndef __BIRI_RTFBO_HPP__
#define __BIRI_RTFBO_HPP__

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <GL/glew.h>
#include "rtshaders.hpp"

void checkFramebufferStatus();

bool isGLExtensionSupported(const char* ext, bool v);

bool initFBO();

#endif
