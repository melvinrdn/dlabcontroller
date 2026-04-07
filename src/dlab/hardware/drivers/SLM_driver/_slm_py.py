import ctypes as ct
import sys

if sys.platform == "win32":
    import dlab.hardware.drivers.SLM_driver._slm_win as dll
else:
    # Linux dev stub: no Santec hardware available.
    class _DllStub:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                print(f"[slm stub] {name}({args}, {kwargs})")
                return 0  # SLM_OK
            return _noop
    dll = _DllStub()
    
import numpy as np
import warnings
from os.path import exists

flags = 0x20000000   # Rate 120 Hz SLM

def SLM_STATUS(slm_status):
    """Used to check the return value of every call of the SLM driver,
        should be equal to SLM_OK = 0.
        If that is not the case, error is raised with given error code."""
    
    error_dict = {1: 'SLM_NG', 2: 'SLM_BS', 3: 'SLM_ER',
                  -1: 'SLM_INVALID_MONITOR', -2: 'SLM_NOT_OPEN_MONITOR',
                  -3: 'SLM_OPEN_WINDOW_ERR', -4: 'SLM_DATA_FORMAT_ERR',
                  -101: 'SLM_FILE_READ_ERR', -200: 'SLM_NOT_OPEN_USB',
                  -1000: 'SLM_OTHER_ERROR',
                  -10001: 'FT_INVALID_HANDLE', -10002: 'FT_DEVICE_NOT_FOUND',
                  -10003: 'FT_DEVICE_NOT_OPENED '}
    
    if slm_status == 0:   # SLM_OK
        return
    elif slm_status in error_dict:
        raise RuntimeError('SLM driver failed: ' + error_dict[slm_status] +
                           ', error code ' + str(slm_status))
    raise RuntimeError('SLM driver failed: error code ' + str(slm_status))





def SLM_Disp_Info(display_number):
    """
    Read width and height of the display.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).

    Returns
    -------
    int
        Width of the Display (Pixels).
    int
        Height of the Display (Pixels).
    """
    
    width = ct.c_ushort(0)
    height = ct.c_ushort(0)
    
    ret = dll.SLM_Disp_Info(display_number, width, height)
    SLM_STATUS(ret)
    
    return width.value, height.value



def SLM_Disp_Info2(display_number):
    """
    Read width and height, DisplayName of display.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).

    Returns
    -------
    int
        Width of the Display (Pixels).
    int
        Height of the Display (Pixels).
    str
        DisplayName: DisplayName format is “UserFriendlyName,ManufactreName,ProductCodeID,SerialNumberID”
    """
    
    width = ct.c_ushort(0)
    height = ct.c_ushort(0)
    display_name = ct.create_string_buffer(64)
    
    ret = dll.SLM_Disp_Info2(display_number, width, height, display_name)
    SLM_STATUS(ret)
    
    return width.value, height.value, display_name.value.decode('mbcs')



def SLM_Disp_Open(display_number):
    """
    SLM display initializing.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).

    Returns
    -------
    None.
    """
    
    ret = dll.SLM_Disp_Open(display_number)
    SLM_STATUS(ret)
    
    return



def SLM_Disp_Close(display_number):
    """
    SLM display finalizing.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).

    Returns
    -------
    None.
    """
    
    ret = dll.SLM_Disp_Close(display_number)
    
    if ret == -2:
        warnings.warn('The display is already closed.', RuntimeWarning, 2)
    else:
        SLM_STATUS(ret)
    
    return



def SLM_Disp_GrayScale(display_number, gray_scale):
    """


    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).
    gray_scale : int
        Specify grayscale from 0 to 1023 (0π - 2π).

    Returns
    -------
    None.
    """
    
    ret = dll.SLM_Disp_GrayScale(display_number, flags, gray_scale)
    SLM_STATUS(ret)
    
    return



def SLM_Disp_Data(display_number, data_array, width, height):
    """
    Display array data on the SLM.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).
    data_array : np.array of int
        Data to be displayed.
        Must be 2d array of size (height * width).
        Contains pixel data from 0 to 1023.
    int
        Width of the Display (Pixels).
    int
        Height of the Display (Pixels).

    Returns
    -------
    None.
    """
    
    # check dimensions
    if np.shape(data_array) != (height, width):
        raise IndexError('Array dimensions must match SLM screen resolution of ' 
                         + str(height) + 'x' + str(width) + '.')

    # convert from numpy array to pointer to array of ushort
    data = data_array.astype(np.ushort)
    c = data.ctypes.data_as(ct.POINTER((ct.c_ushort * height) * width)).contents

    # display on SLM
    ret = dll.SLM_Disp_Data(display_number, width, height, flags, c)
    SLM_STATUS(ret)
    
    return



def SLM_Disp_ReadBMP(display_number, path):
    """
    Display .bmp-file data on the SLM.
    Encoding is Unicode.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).
    path : str
        Unicode string containing path to the .bmp-file.
        (Including filename + ".bmp")

    Returns
    -------
    None.
    """
    
    if exists(path):
        ret = dll.SLM_Disp_ReadBMP(display_number, flags, path)
        SLM_STATUS(ret)
    else:
        raise FileNotFoundError('There is no such file in the given directory.')
    
    return



def SLM_Disp_ReadCSV(display_number, path):
    """
    Display .csv-file data on the SLM.
    Encoding is Unicode.

    Parameters
    ----------
    display_number : int
        Specify display number (1, 2, 3…).
    path : str
        Unicode string containing path to the .csv-file.
        (Including filename + ".csv")

    Returns
    -------
    None.
    """
    
    if exists(path):
        ret = dll.SLM_Disp_ReadCSV(display_number, flags, path)
        SLM_STATUS(ret)
    else:
        raise FileNotFoundError('There is no such file in the given directory.')
    
    return


def SLM_Ctrl_Open(slm_number):
    """
    Open USB interface.

    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    Returns
    -------
    None.
    """
    
    ret = dll.SLM_Ctrl_Open(slm_number)
    
    if ret == -10003:
        warnings.warn('The USB interface is already open.', RuntimeWarning, 2)
    else:
        SLM_STATUS(ret)
    
    return



def SLM_Ctrl_Close(slm_number):
    """
    Close USB interface.

    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    Returns
    -------
    None.
    """
    
    ret = dll.SLM_Ctrl_Close(slm_number)
    
    if ret == -200:
        warnings.warn('The USB interface is already closed.', RuntimeWarning, 2)
    else:
        SLM_STATUS(ret)
    
    return



def SLM_Ctrl_ReadSU(slm_number):   #1
    """
    Read status of SLM. Busy or Ready.

    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    Returns
    -------
    bool
        True: SLM ready
        False: SLM busy
    """
    
    ret = dll.SLM_Ctrl_ReadSU(slm_number)
    
    if ret == 0:    return True    # SLM ready
    elif ret == 2:  return False   # SLM busy
    else:           SLM_STATUS(ret)



def SLM_Ctrl_WriteVI(slm_number, mode=1):   #2
    """
    Write video mode: DVI or Memory mode.


    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    mode : int or str
        Specify mode value.
        0: Memory mode (Memory, USB, mem),
        1: DVI mode (DVI),
        Default: 1

    -------
    None.
    """
    
    # read eventual text input
    if mode in [0, 'Memory', 'memory', 'USB', 'usb', 'mem', 'Memory mode', 'M', 'm']:
        mode = 0
    elif mode in [1, 'DVI', 'dvi', 'DVI mode', 'D', 'd']:
        mode = 1
    else:
        raise ValueError('Please specify a valid mode: "Memory" (0) or "DVI" (1).')
    
    # change mode
    ret = dll.SLM_Ctrl_WriteVI(slm_number, mode)
    SLM_STATUS(ret)
    
    return



def SLM_Ctrl_ReadVI(slm_number):   #3
    """
    Write video mode: DVI or Memory mode.


    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    -------
    int
        Current operating mode mode.
        0: Memory mode,
        1: DVI mode.
    """

    mode = ct.c_uint32(0)

    ret = dll.SLM_Ctrl_ReadVI(slm_number, mode)
    SLM_STATUS(ret)

    return mode.value



def SLM_Ctrl_WriteWL(slm_number, wavelength, phase):
    
    ret = dll.SLM_Ctrl_WriteWL(slm_number, wavelength, phase)
    SLM_STATUS(ret)
    
    return



def SLM_Ctrl_ReadWL(slm_number):
    
    wavelength = ct.c_uint32(0)
    phase = ct.c_uint32(0)
    
    ret = dll.SLM_Ctrl_ReadWL(slm_number, wavelength, phase)
    SLM_STATUS(ret)
    
    return wavelength.value, phase.value



def SLM_Ctrl_ReadT(slm_number):   #35
    """
    Read drive board and option board Celsius temperatures.

    Parameters
    ----------
    slm_number : int
        Specify SLM number (1-8).

    Returns
    -------
    float
        Driveboard temperature in °C.
    float
        Optionboard temperature in °C.
    """
    
    T1 = ct.c_ulong(0)
    T2 = ct.c_ulong(0)
    
    ret = dll.SLM_Ctrl_ReadT(1, T1, T2)
    SLM_STATUS(ret)
    
    return T1.value/10, T2.value/10
