#!/bin/bash

# Build and Test Script for sparse-maq C++ Library
# This script compiles all test files and runs them

set -e  # Exit on first error

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  sparse-maq C++ Build & Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create build directory if it doesn't exist
if [ ! -d "core/build" ]; then
  echo -e "${BLUE}Creating build directory...${NC}"
  mkdir -p core/build
fi

# Compiler settings
CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -Wall -Wextra -I core/src"

echo -e "${BLUE}Compiler: ${CXX}${NC}"
echo -e "${BLUE}Flags: ${CXXFLAGS}${NC}"
echo ""

# Test files to compile
tests=(
  "test_preprocess_data"
  "test_convex_hull"
  "test_compute_path"
  "test_e2e"
)

# Compile each test
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Compiling Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for test in "${tests[@]}"; do
  echo -e "${BLUE}Compiling ${test}...${NC}"
  ${CXX} ${CXXFLAGS} -o "core/build/${test}" "core/tests/${test}.cpp"
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ ${test} compiled successfully${NC}"
  else
    echo -e "${RED}✗ ${test} compilation failed${NC}"
    exit 1
  fi
  echo ""
done

# Run each test
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Running Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

failed_tests=()
passed_tests=()

for test in "${tests[@]}"; do
  echo -e "${BLUE}Running ${test}...${NC}"
  echo -e "${BLUE}----------------------------------------${NC}"

  if "./core/build/${test}"; then
    echo -e "${GREEN}✓ ${test} passed${NC}"
    passed_tests+=("${test}")
  else
    echo -e "${RED}✗ ${test} failed${NC}"
    failed_tests+=("${test}")
  fi
  echo ""
done

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

total_tests=${#tests[@]}
passed_count=${#passed_tests[@]}
failed_count=${#failed_tests[@]}

echo -e "Total tests: ${total_tests}"
echo -e "${GREEN}Passed: ${passed_count}${NC}"

if [ ${failed_count} -gt 0 ]; then
  echo -e "${RED}Failed: ${failed_count}${NC}"
  echo ""
  echo -e "${RED}Failed tests:${NC}"
  for test in "${failed_tests[@]}"; do
    echo -e "${RED}  - ${test}${NC}"
  done
  echo ""
  exit 1
else
  echo -e "${RED}Failed: ${failed_count}${NC}"
  echo ""
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}  All tests passed! 🎉${NC}"
  echo -e "${GREEN}========================================${NC}"
  exit 0
fi
