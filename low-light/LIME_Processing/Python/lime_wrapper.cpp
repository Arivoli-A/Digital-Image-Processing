#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <opencv2/opencv.hpp>
#include "lime.h"

namespace py = pybind11;
using namespace feature;

// ------------------------------------------------------------
// NumPy <-> cv::Mat converters
// ------------------------------------------------------------
static cv::Mat numpy_uint8_3c_to_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 3 || buf.shape[2] != 3)
        throw std::runtime_error("Expected HxWx3 uint8 image");

    cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    return mat.clone();   // clone to own memory
}

static py::array_t<uint8_t> mat_to_numpy_uint8_3c(const cv::Mat& mat) {
    if (mat.type() != CV_8UC3)
        throw std::runtime_error("Expected CV_8UC3 output");

    auto result = py::array_t<uint8_t>({ mat.rows, mat.cols, 3 });
    memcpy(result.mutable_data(), mat.data, mat.total() * mat.elemSize());
    return result;
}

// ------------------------------------------------------------
// PYBIND11 MODULE
// ------------------------------------------------------------
PYBIND11_MODULE(pylime, m) {
    m.doc() = "Python bindings for feature::lime class";

    py::class_<lime>(m, "lime")
        // constructor: lime(cv::Mat src)
        .def(py::init([](py::array_t<uint8_t> src) {
            cv::Mat mat = numpy_uint8_3c_to_mat(src);
            return new lime(mat);
        }))

        // method: cv::Mat lime_enhance(cv::Mat& src)
        .def("lime_enhance", [](lime& self, py::array_t<uint8_t> src) {
            cv::Mat mat = numpy_uint8_3c_to_mat(src);
            cv::Mat out = self.lime_enhance(mat);
            return mat_to_numpy_uint8_3c(out);
        })

        // method: Illumination
        .def("Illumination", [](lime& self,
                                py::array_t<uint8_t> src) {
            cv::Mat mat = numpy_uint8_3c_to_mat(src);
            cv::Mat out;
            self.Illumination(mat, out);
            return mat_to_numpy_uint8_3c(out);
        })

        // method: Illumination_filter
        .def("Illumination_filter", [](lime& self,
                                       py::array_t<uint8_t> src) {
            cv::Mat img = numpy_uint8_3c_to_mat(src);
            cv::Mat out;
            self.Illumination_filter(img, out);
            return mat_to_numpy_uint8_3c(out);
        })

        // expose data members
        .def_readwrite("channel", &lime::channel)
        .def_readwrite("out_lime", &lime::out_lime);  // rarely needed
}
