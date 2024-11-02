/*
 * clip2tri.h
 *
 * Authors: kaen, raptor, sam686, watusimoto
 *
 * Originally from the bitfighter source code
 */

#ifndef CLIP2TRI_H_
#define CLIP2TRI_H_

#include <vector>

#include "../clipper/clipper.hpp"

using namespace std;
using namespace ClipperLib;

namespace c2t
{

typedef signed int       S32;
typedef signed long long S64;
typedef unsigned int     U32;
typedef float            F32;
typedef double           F64;


struct Point
{
   F32 x;
   F32 y;

   Point();
   Point(const Point &pt);

   template<class T, class U>
   Point(T in_x, U in_y) { x = static_cast<F32>(in_x); y = static_cast<F32>(in_y); }
};


class clip2tri
{
private:
   //
   Path upscaleClipperPoints(const vector<Point> &inputPolygon);

   // These operate on a vector of polygons
   Paths upscaleClipperPoints(const vector<vector<Point> > &inputPolygons);
   vector<vector<Point> > downscaleClipperPoints(const Paths &inputPolygons);

   bool mergePolysToPolyTree(const vector<vector<Point> > &inputPolygons, PolyTree &solution);

   bool triangulateComplex(vector<Point> &outputTriangles, const Path &outline,
         const PolyTree &polyTree, bool ignoreFills = true, bool ignoreHoles = false);

public:
   clip2tri();
   virtual ~clip2tri();

   void triangulate(const vector<vector<Point> > inputPolygons, vector<Point> &outputTriangles,
         const vector<Point> boundingPolygon);
};

} /* namespace c2t */

#endif /* CLIP2TRI_H_ */
