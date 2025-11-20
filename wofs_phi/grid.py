#======================================================
# This module handles the Grid-related attributes. 
# For WoFS-PHI, we will primarily be dealing with the
# WoFS grid, but that could change if other -PHI systems
# are developed (e.g., HRRR-PHI). This code is designed
# to be flexible. 
#=======================================================

class Grid: 
    """ Defines a forecast grid for ML, plotting, etc."""

    def __init__(self, ny, nx, lats, lons, tlat1, tlat2, stlon, sw_lat, ne_lat, sw_lon, ne_lon, ypts, xpts):
        """
            @ny is number of y grid points,
            @nx is number of x grid points,
            @lats is list of latitudes 
            @lons is list of longitudes
            @tlat1 is true latitude 1
            @tlat2 is true latitude 2
            @stlon is standard longitude
            @sw_lat is the southwest corner latitude
            @ne_lat is the northeast corner latitude
            @sw_lon is the southwest corner longitude
            @ne_lon is the northeast corner longitude 
            @ypts is a 2-d array of y values (ny,nx) 
            @xpts is a 2-d array of x values (ny,nx) 
        """

        self.ny = ny
        self.nx = nx
        self.lats = lats
        self.lons = lons
        self.tlat1 = tlat1
        self.tlat2 = tlat2
        self.stlon = stlon
        self.sw_lat = sw_lat
        self.ne_lat = ne_lat
        self.sw_lon = sw_lon
        self.ne_lon = ne_lon
        self.ypts = ypts
        self.xpts = xpts


        return 


    #TODO: Not finished with this. 
    @classmethod
    def create_wofs_grid(cls, wofs_path, wofs_file):
        '''Creates a Grid object from a wofs path and wofs file.'''

        full_wofs_file = "%s/%s" %(wofs_path, wofs_file)

        #Get legacy file
        legacy_fnames = WoFS_Agg.get_legacy_filenames("mslp", [wofs_file])
        legacy_fname = legacy_fnames[0]

        full_legacy_wofs_file = "%s/%s" %(wofs_path, legacy_fname)

        #Add capabilities to account for ALL or legacy file names
        if (c.use_ALL_files == True):
            try:
                ds = open_dataset(full_wofs_file, decode_times=False)
            except FileNotFoundError:
                try:
                    ds = open_dataset(full_legacy_wofs_file, decode_times=False)
                except:
                    print ("Neither %s nor %s found" %(full_wofs_file, full_legacy_wofs_file))
                    quit()

        else:

            try:
                ds = open_dataset(full_legacy_wofs_file, decode_times=False)
            except FileNotFoundError:
                try:
                    ds = open_dataset(full_wofs_file, decode_times=False)
                except:
                    print ("Neither %s nor %s found" \
                            %(full_legacy_wofs_file, full_wofs_file))
                    quit()


        ny = int(ds.attrs['ny'])
        nx = int(ds.attrs['nx'])

        wofsLats = ds.variables['xlat'][:]
        wofsLons = ds.variables['xlon'][:]

        Tlat1 = ds.attrs['TRUELAT1']
        Tlat2 = ds.attrs['TRUELAT2']
        Stlon = ds.attrs['STAND_LON']

        SW_lat = wofsLats[0,0]
        NE_lat = wofsLats[-1,-1]
        SW_lon = wofsLons[0,0]
        NE_lon = wofsLons[-1,-1]

        #Find arrays of x and y points 
        xArr, yArr = get_xy_points(ny, nx)

        #Create new wofs Grid object 
        wofs_grid = cls(ny, nx, wofsLats, wofsLons, Tlat1, Tlat2, Stlon, SW_lat, NE_lat, SW_lon, NE_lon, yArr, xArr)

        return wofs_grid



#Global methods 

def get_xy_points(num_y, num_x):
        """Returns 2-grids of x and y points.
            @num_y : int : Number of points in the y-direction
            @num_x : int : Number of points in the x-direction  
        """

        x_arr = np.zeros((num_y, num_x))
        y_arr = np.zeros((num_y, num_x))

        for x in range(num_x):
            for y in range(num_y):
                x_arr[y,x] = x
                y_arr[y,x] = y

        return x_arr, y_arr


