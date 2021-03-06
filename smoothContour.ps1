﻿$vrt_file = '.\srtm_l2.vrt'
$results_dir = '.\Contour'
$contour_smoothing = '.\contour_smoothing.py'

# Make temporary folder
if (-not (Test-Path "$results_dir\_wrk")) {
    mkdir "$results_dir\_wrk"
    
    }
 $wrk = "$results_dir\_wrk"   


Function Get-DEM_name ( $x, $y ) {
    if ([int] $x -ge 0) { $lon = 'e' }
    else { $lon = 'w' }
    
    if ([int] $y -ge 0) { $lat = 'n' }
    else { $lat = 's' }
    
    $xa = [math]::abs($x)
    $ya = [math]::abs($y)
    
    $($lon+$xa.ToString("000") + $lat+$ya.ToString("00"))
    }

Function Get-SmoothDEM ( $vrt, [int] $min_x, [int] $min_y, [int] $max_x, [int] $max_y ) {
	# $vrt is a path to a gdal-vrt-file of a global 1 arch-sec dem. 
	
    $DEM_name = Get-DEM_name $min_x $min_y
    
    if (-not (Test-Path "$results_dir\$DEM_name.shp"))
    {

    echo $("$wrk\$DEM_name" + '.tif')

    gdalwarp `
    -te $($min_x-0.1) $($min_y-0.1) $($max_x+0.1) $($max_y+0.1) `
    -srcnodata -32767 `
    -dstnodata -32767 `
    $vrt `
    $("$wrk\$DEM_name" + '.tif')
    
    pkgetmask `
    -i $("$wrk\$DEM_name" + '.tif') `
    -o $("$wrk\$DEM_name" + '_mask.tif') `
    -min -32766 `
    -nodata 0 `
    -data 1
    
    pkfillnodata `
    -i $("$wrk\$DEM_name" + '.tif') `
    -o $("$wrk\$DEM_name" + '_fill.tif') `
    -m $("$wrk\$DEM_name" + '_mask.tif') `
    -d 500
    
    #get resolution
    
    $x = $(pkinfo `
    -ns `
    -i $("$wrk\$DEM_name" + '.tif')) `
    -replace '\D+'
    #$x -as [int]
    
    $y = $(pkinfo `
    -nl `
    -i $("$wrk\$DEM_name" + '.tif')) `
    -replace '\D+'
    #$y -as [int]
    
    #create smooth raster by cubic spline
    gdalwarp `
    -ts $([math]::round($x/6)) $([math]::round($y/6)) `
    -r cubicspline `
    $("$wrk\$DEM_name" + '_fill.tif') `
    $("$wrk\$DEM_name" + '_s.tif')
    
    gdalwarp `
    -ts $x $y `
    -r cubicspline `
    $("$wrk\$DEM_name" + '_s.tif') `
    $("$wrk\$DEM_name" + '_sl.tif')
    
    
    #create TRI map
    gdaldem `
    TRI `
    $("$wrk\$DEM_name" + '_s.tif') `
    $("$wrk\$DEM_name" + '_r.tif')
    
    gdalwarp `
    -ts $x $y `
    -r cubicspline `
    $("$wrk\$DEM_name" + '_r.tif') `
    $("$wrk\$DEM_name" + '_rl.tif') 

    gdal_calc.py `
    -A $("$wrk\$DEM_name" + '_fill.tif') `
    -B $("$wrk\$DEM_name" + '_sl.tif') `
    -C $("$wrk\$DEM_name" + '_rl.tif') `
    --type=Int16 `
    --outfile $("$wrk\$DEM_name" + '_res.tif') `
    --calc " A*(C>20.0) + (C<=20.0)*( (B*( 1-(C/20.0) ))+(A*C/20.0) ) "
    
    #create contour
    gdal_contour `
    -a height `
    -i 20.0 `
    -snodata 32767 `
    $("$wrk\$DEM_name" + '_res.tif') `
    $("$wrk\$DEM_name" + '_f.shp')  
    
    #clip contours
    ogr2ogr `
    -clipsrc $min_x $min_y $max_x $max_y `
    -skipfailures `
    $("$wrk\$DEM_name" + '_clip.shp') `
    $("$wrk\$DEM_name" + '_f.shp')  
    
    #smoothing of contourlines
    python `
    $contour_smoothing `
    $("$wrk\$DEM_name" + '_clip.shp') `
    $("$results_dir\$DEM_name" + '.shp')
    
    #remove temp-files
    Remove-Item "$wrk\$DEM_name*"
    }
    }

# Delete temporary files folder
#Remove-Item $wrk -recurse

#Get-SmoothDEM $vrt_file 13 56 14 57
