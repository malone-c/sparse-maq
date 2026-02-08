#!/bin/bash

# Script to build and run the sparse-maq C++ demo

set -e  # Exit on error

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${BLUE}Building sparse-maq C++ demo...${NC}"

# Create build directory if it doesn't exist
if [ ! -d "core/build" ]; then
  mkdir -p core/build
fi

# Compile with c++20 standard and include path
g++ -std=c++20 -I core/src -o core/build/main_cpp core/src/main.cpp

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✓ Build successful${NC}"
  echo ""
  echo -e "${BLUE}Running demo...${NC}"
  echo ""
  ./core/build/main_cpp
else
  echo -e "${RED}✗ Build failed${NC}"
  exit 1
fi
