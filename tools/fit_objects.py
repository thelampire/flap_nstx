import numpy as np
from numpy.linalg import eig, inv
from scipy.optimize import curve_fit
from functools import cached_property
from skimage.measure import EllipseModel

class FitEllipse:
    def __init__(self,
                 x=None, y=None,
                 method='linalg', 
                 elongation_base='size',
                 verbose=False, test=False):

        valid_methods = ['linalg', 'skimage', 'leastsquare', 'linalg_v0']
        if method not in valid_methods:
            raise ValueError(f"Method '{method}' invalid. Choose from: {valid_methods}")

        if x is None or y is None:
            raise TypeError('x or y is not set.')
        if len(x) != len(y):
            raise ValueError('The length of x and y must be the same.')

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self._elongation_base = elongation_base
        self._verbose = verbose
        self._test = test

        # Vectorized interpolation for polygons with < 6 vertices
        if len(self.x) < 6:
            x_mid = (self.x + np.roll(self.x, -1)) / 2.0
            y_mid = (self.y + np.roll(self.y, -1)) / 2.0
            
            # Instantly interweaves the midpoints between the original points
            indices = np.arange(1, len(self.x) + 1)
            self.x = np.insert(self.x, indices, x_mid)
            self.y = np.insert(self.y, indices, y_mid)

        self._xmean = np.mean(self.x)
        self._ymean = np.mean(self.y)

        # Initialize fitting
        try:
            if method == 'linalg':
                self._fit_ellipse_linalg(self.x, self.y)
            elif method == 'skimage':
                self._fit_ellipse_skimage(self.x, self.y)
            elif method == 'leastsquare':
                self._fit_ellipse_leastsq(self.x, self.y)
            elif method == 'linalg_v0':
                self._fit_ellipse_linalg_v0(self.x, self.y)
        except Exception as e:
            if self._verbose: 
                print(f"Ellipse fitting failed: {e}")
            self.set_invalid()

    def set_invalid(self):
        self._angle = np.nan
        self._axes_length = np.array([np.nan, np.nan])
        self._center = np.array([np.nan, np.nan])
        self._parameters = np.full(6, np.nan)

    # --- GEOMETRIC PROPERTIES ---
    
    @property
    def angle(self): return self._angle

    @property
    def axes_length(self): return self._axes_length

    @property
    def axes(self): return self._axes_length

    @property
    def center(self): return self._center

    @cached_property
    def size(self):
        alfa = self.angle
        a, b = self.axes_length

        if np.isnan(a) or np.isnan(b) or a == 0 or b == 0:
            return np.array([np.nan, np.nan])

        a0 = (np.cos(alfa)**2 / a**2) + (np.sin(alfa)**2 / b**2)
        a2 = (np.sin(alfa)**2 / a**2) + (np.cos(alfa)**2 / b**2)

        xsize = 2 / np.sqrt(a0)
        ysize = 2 / np.sqrt(a2)

        if np.iscomplex(xsize) or np.iscomplex(ysize):
            if self._verbose: print('Size is complex')
            return np.array([np.nan, np.nan])

        return np.array([xsize, ysize])

    @cached_property
    def elongation(self):
        if self._elongation_base == 'size':
            s1, s2 = self.size
            return (s1 - s2) / (s1 + s2) if (s1 + s2) != 0 else np.nan
        else: # 'axes'
            a1, a2 = self.axes_length
            return (a1 - a2) / (a1 + a2) if (a1 + a2) != 0 else np.nan

    # --- LINALG FIT ---
    
    def _fit_ellipse_linalg(self, x, y):
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2
        
        # Used np.linalg.solve instead of inv() for numerical stability
        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.solve(C, M)
        
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, con > 0]
        
        self._parameters = np.concatenate((ak, T @ ak)).ravel()
        self._center = self._calculate_center_linalg()
        self._axes_length = self._calculate_axes_length_linalg()
        self._angle = self._calculate_angle_linalg()

    def _calculate_angle_linalg(self):
        a, b, c = self._parameters[0], self._parameters[1]/2, self._parameters[2]
        
        if b == 0:
            phi = 0 if a < c else np.pi/2
        else:
            phi = np.arctan((2.*b) / (a - c)) / 2
            if a > c:
                phi += np.pi/2
                
        if not getattr(self, '_width_gt_height', True):
            phi += np.pi/2
            
        # Modulo trick bounds it efficiently between [-pi/2, pi/2]
        phi = (phi + np.pi/2) % np.pi - np.pi/2
        return phi

    def _calculate_axes_length_linalg(self):
        p = self._parameters
        a, b, c, d, f, g = p[0], p[1]/2, p[2], p[3]/2, p[4]/2, p[5]
        den = b**2 - a*c
        if den > 0:
            raise ValueError('Coeffs do not represent an ellipse: b^2 - 4ac must be negative!')

        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        
        ap = np.sqrt(num / den / (fac - a - c))
        bp = np.sqrt(num / den / (-fac - a - c))

        self._width_gt_height = ap >= bp
        return np.array([bp, ap]) if ap < bp else np.array([ap, bp])

    def _calculate_center_linalg(self):
        p = self._parameters
        a, b, c, d, f = p[0], p[1]/2, p[2], p[3]/2, p[4]/2
        den = b**2 - a*c
        if den > 0:
            raise ValueError('Coeffs do not represent an ellipse.')
        return np.array([(c*d - b*f) / den, (a*f - b*d) / den])

    # --- LEAST SQUARES FIT ---
    
    def _fit_ellipse_leastsq(self, x, y):
        aat = np.array([
            [np.sum(x**4),     np.sum(2 * x**3 * y), np.sum(x**2 * y**2), np.sum(2 * x**3),   np.sum(2 * x**2 * y)],
            [np.sum(2 * x**3 * y), np.sum(4 * x**2 * y**2), np.sum(2 * x * y**3), np.sum(4 * x**2 * y), np.sum(4 * x * y**2)],
            [np.sum(x**2 * y**2),  np.sum(2 * x * y**3),  np.sum(y**4),       np.sum(2 * x * y**2), np.sum(2 * y**3)],
            [np.sum(2 * x**3),     np.sum(4 * x**2 * y),  np.sum(2 * x * y**2), np.sum(4 * x**2),   np.sum(4 * x * y)],
            [np.sum(2 * x**2 * y), np.sum(4 * x * y**2),  np.sum(2 * y**3),     np.sum(4 * x * y),  np.sum(4 * y**2)]
        ])
        coord_vec = np.array([np.sum(x**2), np.sum(2*x*y), np.sum(y**2), np.sum(2*x), np.sum(2*y)])

        # np.linalg.solve is vastly superior to np.matmul(inv(A), B)
        self._parameters = np.linalg.solve(aat, coord_vec)

        self._axes_length = self._calculate_axes_length_leastsq()
        self._angle = self._calculate_angle_leastsq()
        self._center = self._calculate_center_leastsq()

    def _calculate_angle_leastsq(self):
        a, b, c = self._parameters[0:3]
        return np.pi/2 + 0.5 * np.arctan2(2*b, a-c)

    def _calculate_center_leastsq(self):
        a, b, c, d, f = self._parameters[0:5]
        den = b**2 - a*c
        return np.array([(c*d - b*f) / den, (a*f - b*d) / den])

    def _calculate_axes_length_leastsq(self):
        a, b, c, d, f, g = *self._parameters[0:5], -1
        nom = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        term = np.sqrt((a-c)**2 + 4 * b**2)
        
        A = np.sqrt(nom / ((b**2 - a*c) * (term - (a+c))))
        B = np.sqrt(nom / ((b**2 - a*c) * (-term - (a+c))))
        return np.array([A, B])

    # --- SKIMAGE FIT ---
    
    def _fit_ellipse_skimage(self, x, y):
        coordinates = np.column_stack((x.ravel(), y.ravel()))
        try:
            ellipse = EllipseModel()
            ellipse.estimate(coordinates)
            xc, yc, a, b, theta = ellipse.params
        except Exception:
            xc, yc, a, b, theta = [np.nan] * 5

        self._center = np.array([xc, yc])
        if a < b:
            self._axes_length, self._angle = np.array([a, b]), theta
        else:
            self._axes_length, self._angle = np.array([b, a]), theta
            
    def _fit_ellipse_linalg_v0(self,x,y):
        """
        Wrapper class for fitting an Ellipse and returning its important features.
        It uses the least square approximation method combined with a Lagrangian
        minimalization for the Eigenvalues of the problem.
        Source:
            https://stackoverflow.com/questions/39693869/fitting-an-ellipse-to-a-set-of-data-points-in-python/48002645
            Fitzgibbon, Pilu and Fischer in Fitzgibbon, A.W., Pilu, M., and Fischer R.B., Direct least squares fitting of ellipsees,
            Proc. of the 13th Internation Conference on Pattern Recognition, pp 253–257, Vienna, 1996
        Rewritten as an object, np.argmax(np.abs(E)) modified to np.argmax(E).
        """

        print("This version of the ellipse fitting is deprecated \
              because of the 90degree angle fitting issue. \
                  Please use method='linalg' instead of linalg_v0")


        xnew=x-self._xmean
        ynew=y-self._ymean

        xnew = xnew[:,np.newaxis]
        ynew = ynew[:,np.newaxis]

        D = np.hstack((xnew*xnew, xnew*ynew, ynew*ynew, xnew, ynew, np.ones_like(xnew)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        #n = np.argmax(E)

        self._parameters = V[:,n]


        a,b=self._calculate_axes_length_linalg_v0()
        theta=self._calculate_angle_linalg_v0()
        self._center=self._calculate_center_linalg_v0()
        # theta=self._angle

        if a < b:
            self._axes_length=np.asarray([a,b])
            self._angle=np.arcsin(np.sin(theta))
        else:
            self._axes_length=np.asarray([b,a])
            self._angle=np.arcsin(np.sin(theta))+np.pi/2

    def _calculate_angle_linalg_v0(self):
        p=self._parameters
        b,c,_,_,_,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        if b == 0:
            return 0.
        else:
            return np.arctan(2*b/(a-c))/2.

    def _calculate_axes_length_linalg_v0(self):
        p=self._parameters
        b,c,d,f,g,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1=(b*b-a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])

    def _calculate_center_linalg_v0(self):
        p=self._parameters
        b,c,d,f,_,a = p[1]/2, p[2], p[3]/2, p[4]/2, p[5], p[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num+self._xmean
        y0=(a*f-b*d)/num+self._ymean
        return np.array([x0,y0])
    
    
class FitGaussian:
    def __init__(self, x=None, y=None, data=None, verbose=False):
        self._fwhm_to_sigma = 2 * np.sqrt(2 * np.log(2))
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.data = np.asarray(data)
        self._verbose = verbose
        
        self.fit_gaussian(self.x, self.y, self.data)

    def fit_gaussian(self, x, y, data):
        xdata = np.vstack((x.ravel(), y.ravel()))
        
        # Zero-division protection!
        sum_data = np.sum(data)
        safe_sum = sum_data if sum_data != 0 else 1e-10

        initial_guess = [
            data.max(),                                 
            np.sum(x * data) / safe_sum,                
            np.sum(y * data) / safe_sum,                
            (x.max() - x.min()) / 2 / self._fwhm_to_sigma, 
            (y.max() - y.min()) / 2 / self._fwhm_to_sigma, 
            0.,                                         
            np.mean(data)                               
        ]

        try:
            popt, _ = curve_fit(gaussian2D_fit_function, xdata, data, p0=initial_guess)
            popt[5] = np.arcsin(np.sin(popt[5]))
            self.popt = popt
        except Exception as e:
            if self._verbose: print(f'Gaussian fitting failed: {e}')
            self.set_invalid()
            return

        theta = self.popt[5]
        a, b = np.abs(np.array([self.popt[3], self.popt[4]]) * self._fwhm_to_sigma)

        if a < b:
            self._axes_length = np.array([a, b])
            self._angle = np.arcsin(np.sin(theta))
        else:
            self._axes_length = np.array([b, a])
            self._angle = np.arcsin(np.sin(theta - np.pi/2))

        self._center = np.array([self.popt[1], self.popt[2]])

    def set_invalid(self):
        self.popt = np.full(7, np.nan)

    # --- PROPERTIES ---
    
    @property
    def angle(self): return self._angle

    @property
    def axes_length(self): return self._axes_length

    @property
    def center(self): return self._center

    @cached_property
    def center_of_gravity(self):
        safe_sum = np.sum(self.data) if np.sum(self.data) != 0 else 1e-10
        return np.array([np.sum(self.x * self.data) / safe_sum,
                         np.sum(self.y * self.data) / safe_sum])

    @cached_property
    def elongation(self):
        s1, s2 = self.size
        return (s1 - s2) / (s1 + s2) if (s1 + s2) != 0 else np.nan

    @property
    def half_level(self):
        return (self.popt[0] - self.popt[6]) / 2

    @cached_property
    def size(self):
        alfa = self.angle
        a, b = self.axes_length

        if np.isnan(a) or np.isnan(b) or a == 0 or b == 0:
            return np.array([np.nan, np.nan])

        a0 = (np.cos(alfa)**2 / a**2) + (np.sin(alfa)**2 / b**2)
        a2 = (np.sin(alfa)**2 / a**2) + (np.cos(alfa)**2 / b**2)

        return np.array([2 / np.sqrt(a0), 2 / np.sqrt(a2)])


def gaussian2D_fit_function(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = coords
    xo, yo = float(xo), float(yo)
    
    a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
    b = (np.sin(2*theta)) / (4*sigma_x**2) - (np.sin(2*theta)) / (4*sigma_y**2)
    c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)
    
    g = offset + amplitude * np.exp(-(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
    return g.ravel()