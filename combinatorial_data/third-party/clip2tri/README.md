clip2tri
========

Using clipper and poly2tri together for robust triangulation

## About

This code originates from the Bitfighter project (http://bitfighter.org).  

We needed a way to generate fast and robust robot navigation mesh zones.  Triangulation was a part of this and we had originally used the excellent **Triangle** library (https://www.cs.cmu.edu/~quake/triangle.html).  However, we hope to one day get into the Debian repository and so we needed a more-liberally licensed way to do this.  We found the fast **poly2tri** library in hopes that it would do what we need since no other open source library we found performed as well and had as open a license.

We quickly found that poly2tri has some robustness issues which are inherent to the sweep-line algorithm it uses.  Sanitizing our input using the amazing **clipper** library (http://www.angusj.com/delphi/clipper.php) solves most of the issues.

This code ties it all together.

## Details

Bitfighter generates bot nav-mesh zones using the following process (more or less) using floating point data as the input:

1. Aggregate all polygons and holes
2. Feed data to **clipper** and have it return a PolyTree structure
3. Perform some specific sanitization
4. Feed data into **poly2tri**
5. Feed data into a custom 2D version of recastnavigation (https://github.com/memononen/recastnavigation) to generate the bot zones

The code here handles at least steps 2-4, and has been altered to use more standard data types than those used in the Bitfighter code base.

