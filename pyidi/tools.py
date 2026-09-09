import numpy as np

import logging
import logging.handlers


def update_docstring(target_method, doc_method=None, delimiter='---', added_doc=''):
    """
    Update the docstring in target_method with the docstring from doc_method.

    :param target_method: The method that waits for the docstring
    :type target_method: method
    :param doc_method: The method that holds the desired docstring
    :type doc_method: method
    :param delimiter: insert the desired docstring between two delimiters, defaults to '---'
    :type delimiter: str, optional
    """
    docstring = target_method.__doc__.split(delimiter)

    leading_spaces = len(docstring[1].replace('\n', '')) - len(docstring[1].replace('\n', '').lstrip(' '))

    if doc_method is not None:
        if doc_method.__doc__:
            docstring[1] = doc_method.__doc__
        else:
            docstring[1] = '\n' + ' '*leading_spaces + \
                'The selected method does not have a docstring.\n'
    else:
        docstring[1] = added_doc.replace('\n', '\n' + ' '*leading_spaces)

    target_method.__func__.__doc__ = delimiter.join(docstring)


def split_points(points, processes):
    """Split the array of points to different processes.

    :param points: Array of points (2d)
    :type points: numpy array
    :param processes: number of processes
    :type processes: int
    """
    points = np.asarray(points)
    step = points.shape[0]//processes
    rest = points.shape[0]%processes
    points_split = []

    last_point = 0
    for i in range(processes):
        this_step = step
        if i < rest:
            this_step += 1
        points_split.append(points[last_point:last_point+this_step])
        last_point += this_step

    return points_split

# @nb.njit
def get_gradient(image):
    """Fast gradient computation.

    Compute the gradient of image in both directions using central
    difference weights over 3 points.

    !!! WARNING:
    The edges are excluded from the analysis and the returned image
    is smaller then original.

    :param image: 2d numpy array
    :return: gradient in x and y direction
    """
    im1 = image[2:]
    im2 = image[:-2]
    Gy = (im1 - im2)/2

    im1 = image[:, 2:]
    im2 = image[:, :-2]
    Gx = (im1 - im2)/2

    return Gx[1:-1], Gy[:, 1:-1]


def setup_logger(logger_name, level="DEBUG", backup_count=1):
    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Set up a rotating file handler to manage log file size
    file_handler = logging.handlers.RotatingFileHandler(
        f'{logger_name}.log', maxBytes=1024 * 1024, backupCount=backup_count  # 1 MB per file, keep 5 backups
    )
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger



