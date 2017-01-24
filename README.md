# smoothContour

Scripts to create smooth contourlines from SRTM level 2 data.

Dependencies:
gdal, python-gdal, pktools

Usage:
* Create a .vrt file of your terrain data with gdalbuildvrt.
* Edit first lines of smoothContour.ps1 to point to your .vrt file and your output directory.
* source script
* run command: Get-SmoothDEM $vrt_file 13 56 14 57

This will create a shape file with 20m contourlines for area lon 13-14, lat 56-57.

