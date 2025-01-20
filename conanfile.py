import tarfile
import os
import shutil
from conan import ConanFile
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import mkdir, copy, collect_libs

required_conan_version = ">=1.53.0"


class MatrixConan(ConanFile):
    name = "matrix"
    version = "dev"
    options = {
        "with_test": [False, True],
        "with_cuda": [False, "11.1.1", "11.4.1", "12.1.1", "system", "11.8.0"],
        "with_opencv_gpu": [False, True, "Orin"],
        "with_tensorrt": [False, "8.2.5.1", "8.4.1.5", "8.5.3.1"],
    }
    default_options = {
        "with_test": False,
        "with_cuda": "system",
        "with_opencv_gpu": False,
        "with_tensorrt": "8.5.3.1",
    }

    @property
    def _min_cppstd(self):
        return 17

    def configure(self):
        self.options["glog"].shared = True
        self.options["glog"].with_unwind = False
        self.options["proj"].shared = True
        self.options["libtiff"].jpeg = "libjpeg-turbo"

        # specify boost options
        self.options["boost"].without_locale = True
        self.options["boost"].without_log = True
        self.options["boost"].without_test = True
        self.options["boost"].without_stacktrace = True
        self.options["boost"].without_fiber = True

        # specifiy opencv options
        self.options["opencv"].parallel = "openmp"
        self.options["opencv"].shared = True
        self.options["opencv"].with_ffmpeg = False
        self.options["opencv"].with_tiff = False
        self.options["opencv"].with_webp = False
        self.options["opencv"].with_openexr = False
        self.options["opencv"].with_jpeg2000 = False
        self.options["opencv"].with_gtk = False
        self.options["opencv"].with_jpeg = "libjpeg-turbo"
        self.options["opencv"].with_tesseract = False
        self.options["freeimage"].with_jpeg = "libjpeg-turbo"
        self.options["freeimage"].with_png = False
        self.options["freeimage"].with_tiff = False
        self.options["freeimage"].with_raw = False
        self.options["freeimage"].with_openexr = False
        if self.options.with_opencv_gpu != False:
            self.options["opencv"].with_cuda = True
            self.options["opencv"].cudaarithm = True
            self.options["opencv"].dnn = True
            self.options["opencv"].cudaimgproc = True
            self.options["opencv"].cudawarping = True
            self.options["opencv"].cuda_arch_bin = "7.2,7.5,8.6,8.7"
        if self.options.with_cuda != False:
            self.options["tensorrt"].cuda = self.options.with_cuda
        self.options["ceres-solver"].use_glog = True
        self.options["pcl"].with_kdtree = True
        self.options["abseil"].shared = True
        self.options["jsoncpp"].shared = True
        self.options["gtsam"].with_TBB = False


    def requirements(self):
        self.requires("gtest/1.11.0")
        self.requires("llvm-openmp/18.1.8")
        self.requires("asio/1.28.0", override=True)
        self.requires("boost/1.83.0", force=True)
        self.requires("protobuf/3.21.12", force=True)
        self.requires("jsoncpp/1.8.3")
        # self.requires("tinyxml2/9.0.0")
        self.requires("yaml-cpp/0.7.0")
        self.requires("zlib/1.2.13", override=True)
        self.requires("libuuid/1.0.3")
        self.requires("toml11/3.7.0")
        self.requires("eigen/3.4.0", force=True)
        self.requires("ceres-solver/2.1.0")
        self.requires("glog/0.7.1")
        self.requires("grpc/1.50.1")
        self.requires("gtsam/4.1.1")
        self.requires("proj/7.2.1")
        self.requires("libjpeg-turbo/3.0.0", force=True)
        self.requires("pybind11/2.9.1", private=True)
        self.requires("openssl/3.1.2", override=True)
        self.requires("matplotlib-cpp/0.0.1")
        self.requires("wayland-protocols/1.33", override=True)
        self.requires("opencv/4.8.1")