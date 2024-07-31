import tarfile
import os
import shutil
from conan import ConanFile
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import mkdir, copy, collect_libs

required_conan_version = ">=1.53.0"


class DevastatorConan(ConanFile):
    name = "devastator"
    version = "dev"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_test": [True, False],
        "with_hslidar": [True, False],
        "with_innolidar": [True, False],
        "with_rslidar": [True, False],
        "with_colmap": [True, False],
        "with_cuda": [False, "11.1.1", "11.4.1", "12.1.1", "system", "11.8.0"],
        "with_opencv_gpu": [False, True, "Orin"],
        "with_bsnn": [False, True],
        "with_tensorrt": [False, "8.2.5.1", "8.4.1.5", "8.5.3.1"],
        "with_jetson_multimedia_api": [True, False],
        "with_mviz": [True, False],
        "with_hpp_mode": [True, False],
        "with_disable_hdmap_to_utmmode": [True, False],
        "target": "ANY",
        "project": ["wts", "hs", "pts", "master", "none"],
        "product": ["STD", "none"],
        "ros2": ["system", "foxy", "humble"],
        "middleware" : ["mos", "ros"],
    }
    default_options = {
        "with_test": False,
        "with_hslidar": False,
        "with_innolidar": False,
        "with_rslidar": True,
        "with_colmap": False,
        "with_cuda": "system",
        "with_opencv_gpu": False,
        "with_bsnn": False,
        "with_tensorrt": "8.5.3.1",
        "with_jetson_multimedia_api": False,
        "with_mviz": False,
        "with_hpp_mode": False,
        "with_disable_hdmap_to_utmmode": False,
        "target": None,
        "project": "none",
        "product": "none",
        "ros2": "system",
        "middleware": "ros"
    }

    @property
    def _min_cppstd(self):
        return 14

    def imports(self):
        self.copy("*.so*", src="@libdirs", dst="install/third_party", keep_path=True)

    def configure(self):
        if self.options.middleware == "ros":
            if self.options.ros2 == "foxy" and self.options.product == "none":
                self.options["ros2"].with_rviz = True
                self.options["ros2"].with_rqt = True
                self.options["qt"].with_libjpeg = True
        self.options["glog"].shared = True
        self.options["glog"].with_unwind = False
        self.options["proj"].shared = True
        if self.options.with_mviz:
            self.options["assimp"].shared = True
        self.options["libtiff"].jpeg = "libjpeg-turbo"

        # specify boost options
        self.options["boost"].without_locale = True
        self.options["boost"].without_log = True
        self.options["boost"].without_test = True
        self.options["boost"].without_stacktrace = True
        self.options["boost"].without_fiber = True

        # specifiy opencv options
        #self.options["opencv"].parallel = "openmp"
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
            self.options["opencv"].cuda_arch_bin = "7.2,7.5,8.6"
            if self.options.with_opencv_gpu == "Orin":
                self.options[
                    "opencv"
                ].cuda_arch_bin += ",8.7"  # for Jetson AGX Orin and Drive AGX Orin only
        if self.options.with_cuda != False:
            self.options["tensorrt"].cuda = self.options.with_cuda
        self.options["ceres-solver"].use_glog = True
        self.options["pcl"].with_kdtree = True
        self.options["abseil"].shared = True
        if self.options.product == "STD":
            self.options["foxglove-websocket"].use_a1000 = True
            self.options["jsoncpp"].shared = True


    def requirements(self):
        self.requires("gtest/1.11.0")
        self.requires("asio/1.28.0", override=True)
        self.requires("boost/1.83.0", force=True)
        self.requires("fast-dds/2.3.4@transformer/stable", private=True)
        self.requires("protobuf/3.21.12", force=True)
        self.requires("jsoncpp/1.8.3")
        self.requires("readerwriterqueue/1.0.6")
        self.requires("spdlog/1.9.2")
        self.requires("sqlite3/3.42.0")
        self.requires("tinyxml2/9.0.0")
        self.requires("yaml-cpp/0.7.0")
        self.requires("zlib/1.2.13", override=True)

        self.requires("libuuid/1.0.3")
        #self.requires("nanoflann/1.4.3")
        self.requires("nlohmann_json/3.10.5")
        self.requires("opencv/4.5.5@transformer/stable")
        self.requires("toml11/3.7.0")
        self.requires("eigen/3.4.0", force=True)
        self.requires("xz_utils/5.4.2")

        if self.options.with_cuda:
            self.requires(f"cudatoolkit/{self.options.with_cuda}")
        if self.options.with_bsnn:
            self.requires("bsnn/3.10.1")
            self.requires("rcall/0.0.2")
        if self.options.with_tensorrt and self.options.with_cuda != False:
            if self.options.with_cuda == "11.8.0":
                self.requires(f"tensorrt/{self.options.with_tensorrt}@jinglicheng/4080")
            else:
                self.requires(f"tensorrt/{self.options.with_tensorrt}")
        if self.options.with_jetson_multimedia_api:
            self.requires("jetson_multimedia_api/35.1.0")

        self.requires("adrss/1.1.0")
        #self.requires("adolc/2.7.2")
        self.requires("alc/1.0.7")
        self.requires("ceres-solver/2.1.0")
        self.requires("glog/0.6.0@transformer/stable")
        self.requires("grpc/1.50.1")
        self.requires("gtsam/4.1.1")
        self.requires("proj/7.2.1")
        self.requires("libkml/1.3.0")
        self.requires("osqp/1.0.0-alpha")
        self.requires("abseil/20220623.0")
        self.requires("libjpeg-turbo/3.0.0", force=True)
        self.requires("pybind11/2.9.1", private=True)
        self.requires("pcl/1.11.1")
        # try to solve conflicts
        self.requires("openssl/3.1.2", override=True)
        self.requires("cereal/1.3.1")
        self.requires("acados/0.1.9")
        self.requires("casadi/3.5.5")
        self.requires("ecos/dev")
        self.requires("matplotlib-cpp/0.0.1")
        self.requires("calCks/1.0.0")
        self.requires("mach_mcap/1.3.0")
        self.requires("ald/0.1.8")
    
        if self.options.with_hslidar:
            self.requires("hslidar_sdk/4.3.0")
        if self.options.with_innolidar:
            self.requires("inno_lidar/2.5.0")
        if self.options.with_rslidar:
            self.requires("rslidar_sdk/1.5.10")

        self.requires("foxglove-websocket/1.1.0")

        if self.options.product == "STD":
            self.requires("zeromq/4.3.4")
            self.requires("hiredis/1.0.2")
            self.requires("fast-cdr/1.0.26", force=True)
            self.requires("bstos-codec/0.0.3")
            self.requires("bstos_camera/0.0.3")
            if self.options.middleware == "ros":
                if self.options.ros2 == "foxy":
                    self.requires("zstd/1.5.2")
                    self.requires("libpng/1.6.39")
                    self.requires("libcurl/7.80.0")
                    self.requires("fast-cdr/1.0.26")
                    # self.requires("xkbcommon/1.6.0")
                    self.requires("freetype/2.12.1")
                    # self.requires("fast-dds/2.3.4@transformer/stable")
                    self.requires("foonathan-memory/0.7.3@transformer/stable")
                    self.requires("ros2/foxy", transitive_headers=True, transitive_libs=True)
        else:
            self.requires("cereal/1.3.1")
            self.requires("gflags/2.2.2")
            self.requires("tbb/2020.3")
            self.requires("uwebsockets/20.60.0")

            if self.options.with_mviz:
                self.requires("libtiff/4.5.1", override=True)
                self.requires("assimp/5.2.2")
            if self.options.with_colmap:
                self.requires("colmap/3.8.0")



    def build_requirements(self):
        if self.options.product == "STD":
            self.build_requires("grpc/1.50.1")
            self.build_requires("protobuf/3.21.12")



    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"
        if self.options["opencv"].with_cuda:
            tc.variables["USE_CUDA"] = True
        if self.options.with_test:
            tc.variables["BUILD_TESTING"] = True
        else:
            tc.variables["BUILD_TESTING"] = False
        if self.options.with_colmap:
            tc.variables["BUILD_VISUAL_RELOC"] = True
        else:
            tc.variables["BUILD_VISUAL_RELOC"] = False
        if self.options.with_hpp_mode:
            tc.variables["HPP_MODE"] = True
        if self.options.with_disable_hdmap_to_utmmode:
            tc.variables["DISABLE_HDMAP_TO_UTM"] = True
        tc.variables["CMAKE_EXPORT_COMPILE_COMMANDS"] = True
        tc.generate()
        tc = CMakeDeps(self)
        tc.generate()
        tc = VirtualRunEnv(self)
        tc.generate()
        tc = VirtualBuildEnv(self)
        tc.generate(scope="build")

    def layout(self):
        self.folders.source = "."
        self.folders.build = "build"
        self.folders.generators = self.folders.build

        self.cpp.build.libdirs = ["."]
        self.cpp.build.bindirs = ["bin"]

        self.cpp.package.includedirs = ["include"]

    def build(self):
        cmake = CMake(self)
        cmake.configure(variables={"MACH_PROJECT":str(self.options.project), "MACH_PRODUCT":str(self.options.product)})
        if self.options.target:
            cmake.build(target=str(self.options.target))
        else:
            cmake.build()

    def package(self):
        if self.options.target:
            copy(
                self,
                "*.so*",
                src=self.build_folder,
                dst=self.package_folder + "/lib",
                keep_path=False,
            )
            copy(
                self,
                "*.a",
                src=self.build_folder,
                dst=self.package_folder + "/lib",
                keep_path=False,
            )
        else:
            cmake = CMake(self)
            cmake.install()  # cmake.install() can only work for all target

        copy(self, "*", src=self.package_folder + "/lib", dst=self.source_folder + "/install/lib")
        copy(
            self,
            "*.h*",
            src=self.source_folder + "/src",
            dst=self.package_folder + "/include",
            keep_path=True,
            excludes="ros2",
        )
        copy(
            self,
            "*.pb.h",
            src=self.build_folder + "/src",
            dst=self.package_folder + "/include",
            keep_path=True,
            excludes="ros2",
        )
        # add this because when building single target, includedir wouldn't add those proto dirs, thus package_info's includedirs would go wrong
        if self.options.target:
            mkdir(self, self.package_folder + "/include/deva/local_map/proto")
            mkdir(self, self.package_folder + "/include/deva/planning/apollo/apollo_cyber/cyber/proto")
            mkdir(self, self.package_folder + "/include/deva/planning/apollo/apollo_proto/proto")
            mkdir(self, self.package_folder + "/include/deva/perception_map/avp_perception_map/avp_perception_map_proto/proto")
            mkdir(self, self.package_folder + "/include/common/mos_proto")


    def package_info(self):
        self.cpp_info.libdirs = ["lib"]
        self.cpp_info.includedirs = [
            "include",
            "include/deva/control/navi/impl/include",
            "include/deva/control/navi/impl",
            "include/deva/track/pilot_tracker/include",
            "include/deva/prediction/pilot_pred/include",
            "include/deva/local_map/hdmap/include",
            "include/deva/local_map/proto",
            "include/deva/local_map/math/include",
            "include/deva/perception/bevlane_perceptor/include",
            "include/deva/planning/pilot",
            "include/deva/planning/pilot/impl/planning/include",
            "include/deva/planning/pilot/impl/planning/include/planning",
            "include/deva/planning/pilot/impl/third_party/include",
            "include/deva/planning/pilot/impl/pilot_struct_transformation/include",
            "include/deva/planning/navigation_planning/impl/avp_planning/include",
            "include/deva/planning/apollo/apollo_proto",
            "include/deva/planning/apollo/apollo_cyber",
            "include/deva/planning/apollo/apollo_cyber/include",
            "include/deva/planning/apollo/apollo_common/status/include",
            "include/deva/planning/apollo/apollo_common/vehicle_model/include",
            "include/deva/planning/apollo/apollo_common/vehicle_state/include",
            "include/deva/planning/apollo/apollo_common/math/include",
            "include/deva/planning/apollo/apollo_common/string_util/include",
            "include/deva/planning/apollo/apollo_common/util/include",
            "include/deva/planning/apollo/apollo_common/configs/include",
            "include/deva/planning/apollo/apollo_map",
            "include/deva/planning/apollo/apollo_map/include",
            "include/deva/perception/bezier_bevlane_perceptor/include",
            "include/deva/perception_map/avp_perception_map/avp_perception_map_proto",
            "include/common/mos_proto"
        ]
        self.cpp_info.libs = collect_libs(self)
