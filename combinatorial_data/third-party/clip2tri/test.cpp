
#include "clipper/clipper.hpp"
#include "poly2tri/poly2tri.h"
#include "clip2tri/clip2tri.h"

#include <iostream>
#include <cstdio>

using namespace std;
using namespace c2t;


// TODO actually fill out some test data...  this code compiles, but doesn't work!
int main() {
   vector<vector<Point> > inputPolygons;
   vector<Point> outputTriangles;  // Every 3 points is a triangle

   vector<Point> boundingPolygon;

   clip2tri clip2tri;
   clip2tri.triangulate(inputPolygons, outputTriangles, boundingPolygon);
}
