import numpy as np
import time
import datetime
import os
import shutil
import json
import glob
import warnings

import scipy.signal
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import RectBivariateSpline
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import pickle
import rich.progress

from psutil import cpu_count
from .. import tools
from ..video_reader import VideoReader

from .idi_method import IDIMethod
from . import _lk_kernels
from ._lk_kernels import NUMBA_AVAILABLE, nb
from ..progress_bar import progress_bar, rich_progress_bar_setup
try:
    from qtpy.QtWidgets import QApplication
except ImportError:
    QApplication = None

class LucasKanade(IDIMethod):
    """
    Translation identification based on the Lucas-Kanade method using least-squares
    iterative optimization with the Zero Normalized Cross Correlation optimization
    criterium.
    """  
    def configure(
        self, roi_size=(9, 9), pad=2, max_nfev=20,
        tol=1e-8, int_order=3, verbose=1, show_pbar=True,
        processes=1, resume_analysis=False, reference_image=0, frame_range='full',
        use_compiled_kernel=True
    ):
        """
        Displacement identification based on Lucas-Kanade method,
        using iterative least squares optimization of translatory transformation
        parameters to determine image ROI translations.
        
        :param roi_size: (h, w) height and width of the region of interest.
            ROI dimensions should be odd numbers. Defaults to (9, 9)
        :type roi_size: tuple, list, int, optional
        :param pad: size of padding around the region of interest in px, defaults to 2
        :type pad: int, optional
        :param max_nfev: maximum number of iterations in least-squares optimization, 
            defaults to 20
        :type max_nfev: int, optional
        :param tol: tolerance for termination of the iterative optimization loop.
            The minimum value of the optimization parameter vector norm.
        :type tol: float, optional
        :param int_order: interpolation spline order
        :type int_order: int, optional
        :param verbose: show text while running, defaults to 1
        :type verbose: int, optional
        :param show_pbar: show progress bar, defaults to True
        :type show_pbar: bool, optional
        :param processes: number of processes to run
        :type processes: int, optional, defaults to 1.
        :param resume_analysis: if True, the last analysis results are loaded and computation continues from last computed time point.
        :type resum_analysis: bool, optional
        :param reference_image: The reference image for computation. Can be index of a frame, tuple (slice) or numpy.ndarray that
            is taken as a reference.
        :type reference_image: int or tuple or ndarray
        :param frame_range: Part of the video to process. If "full", a full video is processed. If first element of tuple is not 0,
            a appropriate reference image should be chosen.
        :type frame_range: tuple or "full"
        :param use_compiled_kernel: use the compiled (numba) kernel, which fuses the
            inner optimization loop and parallelizes over points. Results are identical
            to the pure NumPy path to within floating-point round-off. Falls back to
            the NumPy path automatically if numba is not installed or if
            ``int_order != 3``. Defaults to True.
        :type use_compiled_kernel: bool, optional
        """
        # The arguments are mapped to the class attributes
        # The class attributes are only overwritten if the argument is not None.
        # This enables the ``get_displacements`` method to update only the necessary
        # parameters, while keeping the rest of the configuration unchanged.
        if pad is not None:
            self.pad = pad
        if max_nfev is not None:
            self.max_nfev = max_nfev
        if tol is not None:
            self.tol = tol
        if verbose is not None:
            self.verbose = verbose
        if show_pbar is not None:
            self.show_pbar = show_pbar
        if roi_size is not None:
            if type(roi_size) is int:
                self.roi_size = [roi_size, roi_size]
            self.roi_size = np.array(roi_size, dtype=int)
        if int_order is not None:
            self.int_order = int_order
        if processes is not None:
            self.processes = processes
        if resume_analysis is not None:
            self.resume_analysis = resume_analysis
        if reference_image is not None:
            self.reference_image = reference_image
        if frame_range is not None:
            self.frame_range = frame_range
        if use_compiled_kernel is not None:
            self.use_compiled_kernel = use_compiled_kernel

        # After the attributes are set, other computation can be carried out.
        self._set_frame_range()


    def _set_frame_range(self):
        """Set the range of the video to be processed.
        """
        self.step_time = 1

        if self.frame_range == 'full':
            self.start_time = 1
            self.stop_time = self.video.N
            
        elif type(self.frame_range) is tuple:
            if len(self.frame_range) >= 2:
                if self.frame_range[0] < self.frame_range[1] and self.frame_range[0] >= 0:
                    self.start_time = self.frame_range[0] + self.step_time
                    
                    if self.frame_range[1] <= self.video.N:
                        self.stop_time = self.frame_range[1]
                    else:
                        raise ValueError(f'frame_range can only go to end of video - up to index {self.video.N}. selected range was: {self.frame_range}')
                else:
                    raise ValueError('Wrong frame_range definition.')

                if len(self.frame_range) == 3:
                    self.step_time = self.frame_range[2]

            else:
                raise Exception('Wrong definition of frame_range.')
        else:
            raise TypeError(f'frame_range must be a tuple of start and stop index or "full" ({type(self.frame_range)}')
            
        self.N_time_points = len(range(self.start_time-self.step_time, self.stop_time, self.step_time))


    def calculate_displacements(self):
        """
        Calculate displacements for set points and roi size.

        kwargs are passed to `configure` method. Pre-set arguments (using configure)
        are NOT changed!
        
        """
        video = self.video

        if self.process_number == 0:
            # Happens only once per analysis
            if self.resume_analysis and self.temp_files_check():
                if self.verbose:
                    print('-- Resuming last analysis ---')
                    print(' ')
            else:
                self.resume_analysis = False
                if self.verbose:
                    print('--- Starting new analysis ---')
                    print(' ')

        if self.processes != 1: # multiprocessing
            if not self.resume_analysis:
                self.create_temp_files(init_multi=True)
            
            self.displacements = multi(self.video, self, self.processes, configuration_keys=self.configuration_keys)

            # Clear the temporary files (only once per analysis)
            self.clear_temp_files()

            # multi() merges the workers' failed_points dicts (remapped to global
            # point indices) into self.failed_points, but per-point warnings raised
            # inside a worker use worker-local indices and may not even reach the
            # console under forkserver/spawn. Summarise with the global indices here.
            self._warn_about_failed_points()
            return

        # For a single process
        self.image_size = (video.image_height, video.image_width)

        if self.resume_analysis:
            self.resume_temp_files()
        else:
            self.displacements = np.zeros((self.points.shape[0], self.N_time_points, 2))
            self.create_temp_files(init_multi=False)

        self.warnings = []

        # Points that could no longer be tracked, keyed by point index. They are
        # NaN in the results from the frame at which they were lost.
        self.failed_points = {}

        # Precomputables
        start_time = time.time()

        if self.verbose:
            t = time.time()
            print('Interpolating the reference image...')

        self._interpolate_reference(video)

        use_compiled_kernel = self._compiled_kernel_available()
        if use_compiled_kernel:
            self._prepare_numba_reference()

        if self.verbose:
            print(f'...done in {time.time() - t:.2f} s')

        # Time iteration.
        len_of_task = len(range(self.start_time, self.stop_time, self.step_time))
        for ii, i in enumerate(progress_bar(self.start_time, self.stop_time, self.step_time)):

            # if resuming analysis and completed points are available, skip those points
            if self.resume_analysis and hasattr(self, "completed_points") and self.completed_points > ii:
                continue

            ii = ii + 1

            # Read the frame once per time step. Reading it inside the loop over
            # points re-decodes the same frame for every point, which is free for
            # memory-mapped formats but costs a full decode per point for video
            # and image files.
            frame = np.asarray(video.get_frame(i))

            if use_compiled_kernel:
                self._optimize_frame_numba(frame, ii, i)
            else:
                self._optimize_frame_numpy(frame, ii, i)

            # temp
            self.temp_disp[:, ii, :] = self.displacements[:, ii, :]
            self.update_log(ii)

            # Update progress bar if multiple processes
            if hasattr(self, "progress") and hasattr(self, "task_id"):
                self.progress[self.task_id] = {"progress": ii + 1, "total": len_of_task}
            # Update progress bar in the GUI
            if QApplication is not None and QApplication.instance() is not None:
                QApplication.processEvents()
                
        del self.temp_disp

        self._warn_about_failed_points()

        if self.verbose:
            full_time = time.time() - start_time
            if full_time > 60:
                full_time_m = full_time//60
                full_time_s = full_time%60
                print(f'Time to complete: {full_time_m:.0f} min, {full_time_s:.1f} s')
            else:
                print(f'Time to complete: {full_time:.1f} s')

        # Clear the temporary files (when multiprocessing is not used)
        if self.process_number == 0:
            self.clear_temp_files()


    def _compiled_kernel_available(self):
        """Decide whether the compiled kernel can be used for this configuration.

        :return: True if the fused numba kernel should be used.
        :rtype: bool
        """
        if not getattr(self, 'use_compiled_kernel', True):
            return False

        if not NUMBA_AVAILABLE:
            if not getattr(self, '_numba_warning_issued', False):
                warnings.warn(
                    'numba is not installed, so the compiled kernel is unavailable. '
                    'Falling back to the NumPy implementation, which is a lot slower. '
                    'Install numba for the fast path.'
                )
                self._numba_warning_issued = True
            return False

        # The kernel implements cubic (de Boor) spline evaluation only.
        if self.int_order != 3:
            if not getattr(self, '_int_order_warning_issued', False):
                warnings.warn(
                    f'The compiled kernel supports int_order=3 only (got {self.int_order}). '
                    'Falling back to the NumPy implementation, which is slower. '
                    'Use int_order=3 for the compiled kernel.'
                )
                self._int_order_warning_issued = True
            return False

        return True

    def _prepare_numba_reference(self):
        """Convert the reference splines into flat arrays for the compiled kernel.

        Every region of interest has the same shape, so all points share the same
        knot vectors and only the spline coefficients differ.
        """
        splines = self.interpolation_splines
        tx, ty, _ = splines[0].tck

        self._nb_tx = np.ascontiguousarray(tx, dtype=np.float64)
        self._nb_ty = np.ascontiguousarray(ty, dtype=np.float64)

        n_cy = len(tx) - self.int_order - 1
        n_cx = len(ty) - self.int_order - 1

        self._nb_coeffs = np.empty((len(splines), n_cy, n_cx), dtype=np.float64)
        for p, spline in enumerate(splines):
            self._nb_coeffs[p] = spline.tck[2].reshape(n_cy, n_cx)

        self._nb_points = np.ascontiguousarray(self.points, dtype=np.int64)
        self._nb_out = np.empty((len(splines), 2), dtype=np.float64)
        self._nb_status = np.empty(len(splines), dtype=np.int64)
        self._nb_clamped = np.empty(len(splines), dtype=np.bool_)
        self._edge_warning_issued = False

    def _displacement_is_sane(self, displacement):
        """Check that a displacement is finite and smaller than the image.

        :param displacement: (dy, dx) in pixels
        :type displacement: array-like of size 2
        :return: False if the value is non-finite or physically impossible
        :rtype: bool
        """
        return bool(
            np.all(np.isfinite(displacement))
            and abs(displacement[0]) <= self.image_size[0]
            and abs(displacement[1]) <= self.image_size[1]
        )

    def _record_failed_point(self, p, frame, status):
        """Record a point that could no longer be tracked.

        The first failure is reported immediately so that long analyses give
        early feedback; the rest are collected into ``failed_points`` and
        summarised once the analysis finishes.

        :param p: index of the point
        :type p: int
        :param frame: frame number at which the point was lost
        :type frame: int
        :param status: one of the ``_lk_kernels.STATUS_*`` constants
        :type status: int
        """
        if p in self.failed_points:
            return

        self.failed_points[p] = {'frame': frame, 'status': status}

        if len(self.failed_points) == 1:
            reason = (
                'the gradient matrix became singular (a flat or edge-only ROI)'
                if status == _lk_kernels.STATUS_SINGULAR else
                'the optimization ran away (a poorly conditioned ROI, for example a '
                'single straight edge with no gradient along it)'
            )
            warnings.warn(
                f'Point {p} (position {self.points[p]}) could not be tracked past frame '
                f'{frame}: {reason}. Its displacements are set to NaN from that frame on '
                f'and the analysis continues. Any further lost points are summarised at '
                f'the end; see the failed_points attribute.'
            )

    def _warn_about_failed_points(self):
        """Summarise the points lost during the analysis."""
        if not self.failed_points:
            return

        indices = sorted(self.failed_points)
        shown = ', '.join(str(p) for p in indices[:10])
        if len(indices) > 10:
            shown += f', ... ({len(indices) - 10} more)'

        warnings.warn(
            f'{len(indices)} of {len(self.points)} points could not be tracked and are '
            f'NaN from the frame at which they were lost: {shown}. Reposition them onto '
            f'features with gradient in both directions, or increase roi_size. '
            f'Per-point detail is in the failed_points attribute.'
        )

    def _optimize_frame_numpy(self, frame, ii, i):
        """Run the reference NumPy optimization for all points of a single frame.

        :param frame: the current frame
        :type frame: ndarray
        :param ii: index into the displacement array
        :type ii: int
        :param i: frame number in the video (for error messages)
        :type i: int
        """
        # Iterate over points.
        for p, point in enumerate(self.points):

            if p in self.failed_points:
                # Already lost at an earlier time step.
                self.displacements[p, ii, :] = np.nan
                continue

            previous = self.displacements[p, ii-1, :]
            if not self._displacement_is_sane(previous):
                # Not caught by the ``failed_points`` check above: a resumed
                # analysis restores displacements from a checkpoint written by an
                # interrupted run, and ``failed_points`` itself is not persisted
                # and is reset above, so a point lost before the checkpoint is no
                # longer known to be lost. np.round(NaN).astype(int) is undefined
                # (e.g. INT64_MIN on x86, 0 on arm64), which would otherwise let
                # the point silently "recover" with finite garbage. Mirrors the
                # same guard the compiled kernel applies to ``previous`` in
                # ``_lk_kernels.optimize_frame``.
                self.displacements[p, ii, :] = np.nan
                self._record_failed_point(p, i, _lk_kernels.STATUS_DIVERGED)
                continue

            # start optimization with previous optimal parameter values
            d_init = np.round(previous).astype(int)
            d_res = previous - d_init

            yslice, xslice = self._padded_slice(point+d_init, self.roi_size, self.image_size, 1)
            G = frame[yslice, xslice]

            try:
                displacements = self.optimize_translations(
                    G=G,
                    F_spline=self.interpolation_splines[p],
                    maxiter=self.max_nfev,
                    tol=self.tol,
                    d_subpixel_init=-d_res,
                    point_index=p,
                    frame=i
                )
            except ValueError:
                self.displacements[p, ii, :] = np.nan
                self._record_failed_point(p, i, _lk_kernels.STATUS_SINGULAR)
                continue

            result = displacements + d_init
            if self._displacement_is_sane(result):
                self.displacements[p, ii, :] = result
            else:
                # The iteration ran away without the gradient matrix ever becoming
                # exactly singular. Left alone this returns silent nonsense.
                self.displacements[p, ii, :] = np.nan
                self._record_failed_point(p, i, _lk_kernels.STATUS_DIVERGED)

    def _optimize_frame_numba(self, frame, ii, i):
        """Run the compiled kernel for all points of a single frame.

        :param frame: the current frame
        :type frame: ndarray
        :param ii: index into the displacement array
        :type ii: int
        :param i: frame number in the video (for error messages)
        :type i: int
        """
        _lk_kernels.optimize_frame(
            frame,
            self._nb_points,
            self._nb_tx,
            self._nb_ty,
            self._nb_coeffs,
            self.displacements[:, ii-1, :],
            self._nb_out,
            self._nb_status,
            self._nb_clamped,
            int(self.roi_size[0]),
            int(self.roi_size[1]),
            1,
            self.max_nfev,
            self.tol,
        )

        if not self._edge_warning_issued and self._nb_clamped.any():
            warnings.warn('Reached image edge. The displacement optimization ' +
                'algorithm may not converge, or selected points might be too close ' +
                'to image border. Please check analysis settings.')
            self._edge_warning_issued = True

        self.displacements[:, ii, :] = self._nb_out

        # A point that fails is lost from this time step onwards. Mark it NaN and
        # carry on with the rest: in a several-hundred-point analysis a handful of
        # badly placed points should not throw away every good one.
        failed = self._nb_status != _lk_kernels.STATUS_OK
        if failed.any():
            self.displacements[failed, ii, :] = np.nan
            for p in np.flatnonzero(failed):
                self._record_failed_point(int(p), i, int(self._nb_status[p]))

    def optimize_translations(self, G, F_spline, maxiter, tol, d_subpixel_init=(0, 0),
                              point_index=None, frame=None):
        """
        Determine the optimal translation parameters to align the current
        image subset `G` with the interpolated reference image subset `F`.

        :param G: the current image subset.
        :type G: array of shape `roi_size`
        :param F_spline: interpolated referencee image subset
        :type F_spline: scipy.interpolate.RectBivariateSpline
        :param maxiter: maximum number of iterations
        :type maxiter: int
        :param tol: convergence criterium
        :type tol: float
        :param d_subpixel_init: initial subpixel displacement guess,
            relative to the integrer position of the image subset `G`
        :type d_init: array-like of size 2, optional, defaults to (0, 0)
        :param point_index: index of the point being processed (for error messages)
        :type point_index: int, optional
        :param frame: frame number being processed (for error messages)
        :type frame: int, optional
        :return: the obtimal subpixel translation parameters of the current
            image, relative to the position of input subset `G`.
        :rtype: array of size 2
        """
        G_float = G.astype(np.float64)
        Gx, Gy = tools.get_gradient(G_float)
        G_float_clipped = G_float[1:-1, 1:-1]

        A_inv = compute_inverse(Gx, Gy)

        if A_inv is None:
            point_info = f"index {point_index}" if point_index is not None else "unknown"
            if point_index is not None and hasattr(self, 'points'):
                point_info += f" (position {self.points[point_index]})"
            frame_info = f"frame {frame}" if frame is not None else "unknown frame"
            raise ValueError(
                f"Degenerate ROI at point {point_info}, {frame_info}. "
                f"The gradient matrix is singular (flat region or single-direction gradient). "
                f"Reposition this point away from uniform or edge-only regions."
            )

        # initialize values
        error = 1.
        displacement = np.array(d_subpixel_init, dtype=np.float64)
        delta = displacement.copy()

        y_f = np.arange(self.roi_size[0], dtype=np.float64)
        x_f = np.arange(self.roi_size[1], dtype=np.float64)

        # optimization loop
        for _ in range(maxiter):
            y_f += delta[0]
            x_f += delta[1]

            F = F_spline(y_f, x_f)
            delta, error = compute_delta(F, G_float_clipped, Gx, Gy, A_inv)

            displacement += delta
            if error < tol:
                return -displacement # roles of F and G are switched

        # max_iter was reached before the convergence criterium
        return -displacement


    def _padded_slice(self, point, roi_size, image_shape, pad=None):
        '''Returns a slice that crops an image around a given ``point`` center, 
        ``roi_size`` and ``pad`` size. If the resulting slice would be out of
        bounds of the image to be sliced (given by ``image_shape``), the
        slice is snifted to be on the image edge and a warning is issued.
        
        :param point: The center point coordiante of the desired ROI.
        :type point: array_like of size 2, (y, x)
        :param roi_size: Size of desired cropped image (y, x).
            type roi_size: array_like of size 2, (h, w)
        :param image_shape: Shape of the image to be sliced, (h, w).
            type image_shape: array_like of size 2, (h, w)
        :param pad: Pad border size in pixels. If None, the video.pad
            attribute is read.
        :type pad: int, optional, defaults to None
        :return crop_slice: tuple (yslice, xslice) to use for image slicing.
        '''

        if pad is None:
            pad = self.pad
        y_, x_ = np.array(point).astype(int)
        h, w = np.array(roi_size).astype(int)

        # Bounds checking
        y = np.clip(y_, h//2+pad, image_shape[0]-(h//2+pad+1))
        x = np.clip(x_, w//2+pad, image_shape[1]-(w//2+pad+1))

        if x != x_ or y != y_:
            warnings.warn('Reached image edge. The displacement optimization ' +
                'algorithm may not converge, or selected points might be too close ' + 
                'to image border. Please check analysis settings.')

        yslice = slice(y-h//2-pad, y+h//2+pad+1)
        xslice = slice(x-w//2-pad, x+w//2+pad+1)
        return yslice, xslice


    def _set_reference_image(self, video: VideoReader, reference_image):
        """Set the reference image.
        """
        if type(reference_image) == int:
            ref = video.get_frame(reference_image).astype(float)

        elif type(reference_image) == tuple:
            if len(reference_image) == 2:
                ref = np.zeros((video.image_height, video.image_width), dtype=float)
                for frame in range(reference_image[0], reference_image[1]):
                    ref += video.get_frame(frame)
                ref /= (reference_image[1] - reference_image[0])
  
        elif type(reference_image) == np.ndarray:
            ref = reference_image

        else:
            raise Exception('reference_image must be index of frame, tuple (slice) or ndarray.')
        
        return ref


    def _interpolate_reference(self, video: VideoReader):
        """
        Interpolate the reference image.

        Each ROI is interpolated in advanced to save computation costs.
        Meshgrid for every ROI (without padding) is also determined here and 
        is later called in every time iteration for every point.
        
        :param video: parent object
        :type video: object
        """
        pad = self.pad
        f = self._set_reference_image(video, self.reference_image)
        splines = []
        for point in self.points:
            yslice, xslice = self._padded_slice(point, self.roi_size, self.image_size, pad)

            spl = RectBivariateSpline(
               x=np.arange(-pad, self.roi_size[0]+pad),
               y=np.arange(-pad, self.roi_size[1]+pad),
               z=f[yslice, xslice],
               kx=self.int_order,
               ky=self.int_order,
               s=0
            )
            splines.append(spl)
        self.interpolation_splines = splines

    
    def show_points(self, figsize=(15, 5), cmap='gray', color='r'):
        """
        Shoe points to be analyzed, together with ROI borders.
        
        :param figsize: matplotlib figure size, defaults to (15, 5)
        :param cmap: matplotlib colormap, defaults to 'gray'
        :param color: marker and border color, defaults to 'r'
        """
        roi_size = self.roi_size

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.video.get_frame(0).astype(float), cmap=cmap)
        ax.scatter(self.points[:, 1],
                   self.points[:, 0], marker='.', color=color)

        for point in self.points:
            roi_border = patches.Rectangle((point - self.roi_size//2 - 0.5)[::-1], self.roi_size[1], self.roi_size[0],
                                            linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(roi_border)

        plt.grid(False)
        plt.show()



def multi(video: VideoReader, idi_method: LucasKanade, processes, configuration_keys: list):
    """
    Splitting the points to multiple processes and creating a
    pool of workers.
    
    :param video: VideoReader object
    :type video: VideoReader
    :param idi_method: IDIMethod object
    :type idi_method: IDIMethod
    :param processes: number of processes to run
    :type processes: int
    :param configuration_keys: list of configuration keys
    :type configuration_keys: list
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    # Where forking is unsafe this returns a context that does not fork.
    mp_context = _lk_kernels.worker_process_context()
    if mp_context is None:
        _warn_if_fork_unsafe()

    if processes < 0:
        processes = cpu_count() + processes
    elif processes == 0:
        raise ValueError('Number of processes must not be zero.')

    points = idi_method.points
    points_split = tools.split_points(points, processes=processes)

    idi_kwargs = {
        'input_file': video.input_file,
        'root': video.root,
    }

    if video.file_format == 'np.ndarray':
        idi_kwargs['input_file'] = video.get_frames() # if the input is np.ndarray, the input_file is the actual data
    

    # Set the parameters that are passed to the configure method
    exclude_keys = ["processes"]
    method_kwargs = dict([(k, idi_method.__dict__.get(k, None)) for k in configuration_keys if k not in exclude_keys])

    # Compile once here rather than once per worker.
    _warm_up_kernels(video, method_kwargs)

    print(f'Computation start: {datetime.datetime.now()}')

    t_start = time.time()

    with rich_progress_bar_setup() as progress:
        futures = []
        with (mp_context or multiprocessing).Manager() as manager:
            # this is the key - we share some state between our 
            # main process and our worker functions
            _progress = manager.dict()

            with ProcessPoolExecutor(max_workers=processes,
                                     mp_context=mp_context) as executor:
                for n in range(0, len(points_split)):  # iterate over the jobs we need to run
                    # set visible false so we don't have a lot of bars all at once:
                    task_id = progress.add_task(f"task {n} ({len(points_split[n])} points)")
                    futures.append(executor.submit(worker, points_split[n], idi_kwargs, method_kwargs, n, _progress, task_id))

                # monitor the progress:
                while sum([future.done() for future in futures]) < len(futures):
                    for task_id, update_data in _progress.items():
                        latest = update_data["progress"]
                        total = update_data["total"]
                        # update the progress bar for this task:
                        progress.update(task_id, completed=latest, total=total+1) # add one for the first frame

                out = []
                for future in futures:
                    out.append(future.result())

                out1 = sorted(out, key=lambda x: x[1])

                # Each worker numbers its points from zero, so shift them back
                # onto the full point list before merging.
                idi_method.failed_points = {}
                offset = 0
                for result in out1:
                    for local, detail in result[2].items():
                        idi_method.failed_points[offset + local] = detail
                    offset += len(result[0])

                out1 = np.concatenate([d[0] for d in out1])

    t = time.time() - t_start
    minutes = t//60
    seconds = t%60
    hours = minutes//60
    minutes = minutes%60
    print(f'Computation duration: {hours:0>2.0f}:{minutes:0>2.0f}:{seconds:.2f}')
    
    return out1


def _warm_up_kernels(video: VideoReader, method_kwargs: dict):
    """Compile the numba kernels once, in the parent process, before forking.

    Without this every worker compiles the same kernel independently on a cold
    cache. Measured at 5-6 s for an analysis that runs in well under a second
    once compiled. It also avoids several processes writing the same cache entry
    at the same time, which is a long-standing open numba bug (numba issue
    #4807, "Cache causes Segmentation Faults when generated in parallel").

    Where the start method is ``fork`` the workers inherit the compiled code
    directly; where it is ``spawn`` they load it from the on-disk cache the
    parent has just written. Either way it is compiled once instead of N times.

    This is only an optimization: if anything here fails the workers simply
    compile the kernels themselves, so all errors are swallowed.

    :param video: the video, used to match the frame dtype the workers will see
    :type video: VideoReader
    :param method_kwargs: the configuration passed on to the workers
    :type method_kwargs: dict
    """
    try:
        roi_size = np.array(method_kwargs.get('roi_size', (9, 9)), dtype=int)
        pad = int(method_kwargs.get('pad', 2))
        int_order = int(method_kwargs.get('int_order', 3))
        tol = float(method_kwargs.get('tol', 1e-8))
        use_compiled = method_kwargs.get('use_compiled_kernel', True) and NUMBA_AVAILABLE

        # The compiled code is specialized on the dtype of the frames, so warm
        # it with a real frame rather than a synthetic one.
        frame = np.asarray(video.get_frame(0))

        if not use_compiled or int_order != 3:
            # The NumPy path still goes through compiled helpers. Mirror the
            # array layouts of the real call so the same specialization is hit.
            subset = frame[:roi_size[0] + 2, :roi_size[1] + 2].astype(np.float64)
            Gx, Gy = tools.get_gradient(subset)
            compute_inverse(Gx, Gy)
            compute_delta(np.zeros(Gx.shape), subset[1:-1, 1:-1], Gx, Gy, np.eye(2))
            return

        grid_y = np.arange(-pad, roi_size[0] + pad)
        grid_x = np.arange(-pad, roi_size[1] + pad)
        spline = RectBivariateSpline(
            x=grid_y, y=grid_x, z=np.zeros((len(grid_y), len(grid_x))),
            kx=int_order, ky=int_order, s=0
        )
        tx, ty, coeffs = spline.tck
        n_cy = len(tx) - int_order - 1
        n_cx = len(ty) - int_order - 1

        centre = [frame.shape[0] // 2, frame.shape[1] // 2]
        # Two points, not one: in the real loop ``previous`` is a slice of the 3D
        # displacement array and is therefore non-contiguous, but with a single
        # point that slice is still contiguous and numba would compile a
        # different, unused specialization.
        points = np.array([centre, centre], dtype=np.int64)
        previous = np.zeros((2, 2, 2))

        _lk_kernels.optimize_frame(
            frame,
            points,
            np.ascontiguousarray(tx, dtype=np.float64),
            np.ascontiguousarray(ty, dtype=np.float64),
            np.ascontiguousarray(np.repeat(coeffs.reshape(1, n_cy, n_cx), 2, axis=0),
                                 dtype=np.float64),
            previous[:, 0, :],
            np.zeros((2, 2)),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.bool_),
            int(roi_size[0]),
            int(roi_size[1]),
            1,
            1,
            tol,
        )
    except Exception:
        # Warming the cache is an optimization, never a requirement.
        pass


# Shared with DirectionalLucasKanade; kept under the original name here because
# it has been importable from this module since 1.4.0.
_warn_if_fork_unsafe = _lk_kernels.warn_if_fork_unsafe


def worker(points, idi_kwargs, method_kwargs, i, progress, task_id):
    """
    A function that is called when for each job in multiprocessing.
    """
    method_kwargs['show_pbar'] = False # use the rich progress bar insted of tqdm

    # Each worker gets a single numba thread. Without this, every process would
    # start a full thread pool of its own and the processes would oversubscribe
    # the CPU, which is slower than either form of parallelism on its own.
    if NUMBA_AVAILABLE:
        _lk_kernels.nb.set_num_threads(1)

    video = VideoReader(**idi_kwargs)
    idi = LucasKanade(video)
    idi.configure(**method_kwargs)
    idi.configure_multiprocessing(i+1, progress, task_id) # configure the multiprocessing settings
    idi.set_points(points)
    displacements = idi.get_displacements(autosave=False)

    # The point indices are local to this worker; the parent maps them back.
    return displacements, i, getattr(idi, 'failed_points', {})


# The two helpers below exist in a compiled and a NumPy flavour. Explicit loops
# are much faster once numba compiles them, but much slower than vectorised NumPy
# when it is not installed, so the right one is bound at import time.


def _inverse_loops(Gx, Gy, tol=1e-10):
    """
    Compute the inverse of the gradient matrix, with explicit loops.

    The singular case is reported through a boolean flag rather than ``None``,
    because numba cannot type a function that returns either an array or
    ``None``.

    :param Gx: x-gradient of the image subset
    :param Gy: y-gradient of the image subset
    :param tol: tolerance for detecting singular matrix
    :return: ``(A_inv, ok)``; ``ok`` is False if the matrix is near-singular
    """
    Gx2 = 0.0
    Gy2 = 0.0
    GxGy = 0.0
    for i in range(Gx.shape[0]):
        for j in range(Gx.shape[1]):
            gx = Gx[i, j]
            gy = Gy[i, j]
            Gx2 += gx * gx
            Gy2 += gy * gy
            GxGy += gx * gy

    det = GxGy**2 - Gx2*Gy2
    A_inv = np.empty((2, 2), dtype=np.float64)
    if abs(det) < tol:
        return A_inv, False  # Near-singular matrix

    A_inv[0, 0] = GxGy / det
    A_inv[0, 1] = -Gx2 / det
    A_inv[1, 0] = -Gy2 / det
    A_inv[1, 1] = GxGy / det

    return A_inv, True


def _inverse_vectorised(Gx, Gy, tol=1e-10):
    """Vectorised counterpart of :func:`_inverse_loops`, for use without numba."""
    Gx2 = np.sum(Gx**2)
    Gy2 = np.sum(Gy**2)
    GxGy = np.sum(Gx * Gy)

    det = GxGy**2 - Gx2*Gy2
    if abs(det) < tol:
        return np.empty((2, 2)), False

    return np.array([[GxGy, -Gx2], [-Gy2, GxGy]]) / det, True


def _delta_loops(F, G, Gx, Gy, A_inv):
    """Least-squares translation update, with explicit loops.

    :return: ``(delta, error)``, the update and its norm
    """
    b0 = 0.0
    b1 = 0.0
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F_G = G[i, j] - F[i, j]
            b0 += Gx[i, j] * F_G
            b1 += Gy[i, j] * F_G

    delta = np.empty(2, dtype=np.float64)
    delta[0] = A_inv[0, 0] * b0 + A_inv[0, 1] * b1
    delta[1] = A_inv[1, 0] * b0 + A_inv[1, 1] * b1

    error = np.sqrt(delta[0]**2 + delta[1]**2)
    return delta, error


def _delta_vectorised(F, G, Gx, Gy, A_inv):
    """Vectorised counterpart of :func:`_delta_loops`, for use without numba."""
    F_G = G - F
    b = np.array([np.sum(Gx*F_G), np.sum(Gy*F_G)])
    delta = np.dot(A_inv, b)

    return delta, np.sqrt(np.sum(delta**2))


if NUMBA_AVAILABLE:
    _compute_inverse = nb.njit(cache=True)(_inverse_loops)
    compute_delta = nb.njit(cache=True)(_delta_loops)
else:                                    # pragma: no cover - environment dependent
    _compute_inverse = _inverse_vectorised
    compute_delta = _delta_vectorised


def compute_inverse(Gx, Gy, tol=1e-10):
    """
    Compute the inverse of the gradient matrix for Lucas-Kanade optimization.

    :param Gx: x-gradient of the image subset
    :param Gy: y-gradient of the image subset
    :param tol: tolerance for detecting singular matrix
    :return: inverse matrix, or None if the matrix is near-singular
    """
    A_inv, ok = _compute_inverse(np.ascontiguousarray(Gx, dtype=np.float64),
                                 np.ascontiguousarray(Gy, dtype=np.float64), tol)
    if not ok:
        return None
    return A_inv


# Backwards-compatible aliases: these helpers have been importable under
# these names since before they were actually compiled.
compute_inverse_numba = compute_inverse
compute_delta_numba = compute_delta
