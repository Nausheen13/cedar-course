#!/bin/bash

# Exit on error
set -e

echo "Starting OpenFOAM setup..."

# Update and install git
apt-get update
apt-get install -y git

# Clone repository
cd /data
echo "Cloning repository..."
git clone https://github.com/Nausheen13/cedar-course.git
cd cedar-course

# Source OpenFOAM
#. /opt/openfoam/openfoam2306/etc/bashrc

# Set up solver
echo "Setting up solver..."
cd $FOAM_SOLVERS
mkdir -p pimpleScalarsFoam
cp -r incompressible/pimpleFoam/* pimpleScalarsFoam/
cp /root/cedar-course/pimpleScalarsFoam/* pimpleScalarsFoam/
cd pimpleScalarsFoam
rm -f pimpleFoam.C

# # Create Make/files
# echo "Creating Make files..."
# echo "pimpleScalarsFoam.C

# EXE = \$(FOAM_APPBIN)/pimpleScalarsFoam" > Make/files

# Compile
echo "Compiling solver..."
wmake
cd

# Set up foam user
echo "Setting up foam user..."
useradd -m foam
chown -R foam:foam /data/cedar-course

echo "Setup complete! To run a case:"
echo "1. Switch to foam user: su - foam"
#echo "2. Source OpenFOAM: . /opt/openfoam/openfoam2306/etc/bashrc"
echo "2. Navigate to your case directory: cd /data/cedar-course/mefenemic-base"
echo "3. Run: pimpleScalarsFoam"
