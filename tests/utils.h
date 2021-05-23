#ifndef TESTS_UTILS_H_
#define TESTS_UTILS_H_

#include "activation.h"
#include "convolution.h"
#include "elementwise.h"
#include "graph.h"
#include "graphdl.h"
#include "pooling.h"
#include "refTensor.h"
#include "reduce.h"

using namespace graphdl::core::layers;

using Vec = std::vector<int>;
using UVec = std::vector<unsigned>;
using HostVec = std::vector<HostTensor>;

template<typename T>
struct PrettyPrinter
{
    static std::string get(const T& t)
    {
        return "<unknownType>";
    }
};

template<>
struct PrettyPrinter<int>
{
    static std::string get(const int& i) { return std::to_string(i); }
};
template<>
struct PrettyPrinter<unsigned>
{
    static std::string get(const unsigned& i) { return std::to_string(i); }
};


template<typename T>
struct PrettyPrinter<std::vector<T>>
{
    using TypeVec = std::vector<T>;

    static std::string get(const TypeVec& vec)
    {
        std::string s = "";
        for (int i = 0; i < static_cast<int>(vec.size()) - 1; ++i)
            s += (PrettyPrinter<T>::get(vec[i]) + "x");
        if (vec.size() > 0)
            s += PrettyPrinter<T>::get(vec[vec.size() - 1]);
        return s;
    }
};

template<>
struct PrettyPrinter<MemoryLocation>
{
    static std::string get(const MemoryLocation& loc)
    {
        if (loc == MemoryLocation::kHOST)
            return "HOST";
        return "DEVICE";
    }
};

template<>
struct PrettyPrinter<ReduceType>
{
    static std::string get(const ReduceType& t)
    {
        switch (t)
        {
        case ReduceType::kSUM: return "SUM";
        case ReduceType::kMAX: return "MAX";
        case ReduceType::kMIN: return "MIN";
        }
        return "unknownReduceType";
    }
};

template<>
struct PrettyPrinter<Elementwise>
{
    static std::string get(const Elementwise& e)
    {
        switch (e)
        {
            case Elementwise::kADD: return "ADD";
            case Elementwise::kSUB: return "SUB";
            case Elementwise::kMUL: return "MUL";
            case Elementwise::kDIV: return "DIV";
        }
        return "unknownElementwise";
    }
};

template<>
struct PrettyPrinter<PoolingType>
{
    static std::string get(const PoolingType& pooling)
    {
        return (pooling == PoolingType::kMAX) ? "MAX" : "AVERAGE";
    }
};

template<>
struct PrettyPrinter<PaddingType>
{
    static std::string get(const PaddingType& padding)
    {
        return (padding == PaddingType::kSAME) ? "SAME" : "VALID";
    }
};

template<>
struct PrettyPrinter<DataFormat>
{
    static std::string get(const DataFormat& format)
    {
        return (format == DataFormat::kNHWC) ? "NHWC" : "NCHW";
    }
};


template<>
struct PrettyPrinter<Activation>
{
    static std::string get(const Activation& a)
    {
        switch (a)
        {
            case Activation::kRELU: return "RELU";
            case Activation::kSIGMOID: return "SIGMOID";
            case Activation::kTANH: return "TANH";
            case Activation::kSQUARE: return "SQUARE";
            case Activation::kABS: return "ABS";
            case Activation::kNEG: return "NEG";
            case Activation::kRECIPROCAL: return "RECIPROCAL";
            case Activation::kLOG: return "LOG";
            case Activation::kSQRT: return "SQRT";
            case Activation::kEXP: return "EXP";
            case Activation::kLEAKY_RELU: return "LEAKY_RELU";
            case Activation::kRELU_6: return "RELU6";
            case Activation::kELU: return "ELU";
            case Activation::kSOFTPLUS: return "SOFTPLUS";
            case Activation::kSOFTSIGN: return "SOFTSIGN";
        }

        return "<unknownActivation>";
    }
};

template <typename T1, typename T2>
struct PrettyPrinter<std::pair<T1, T2>>
{
    static std::string get(const std::pair<T1, T2>& p)
    {
        return PrettyPrinter<T1>::get(p.first) + "__" + PrettyPrinter<T2>::get(p.second);
    };
};

template <int n, typename... Types>
struct TuplePrettyPrintHelper
{
    using T = std::tuple<Types...>;
    const static int N = std::tuple_size_v<T>;
    using TypeElem = typename std::tuple_element<N - n, T>::type;

    static std::string get(const T& t) {
        return PrettyPrinter<TypeElem>::get(std::get<N - n>(t)) + "__" +
               TuplePrettyPrintHelper<n - 1, Types...>::get(t);
    }
};

template <typename... Types>
struct TuplePrettyPrintHelper<1, Types...>
{
    using T = std::tuple<Types...>;
    const static int N = std::tuple_size_v<T>;
    using TypeElem = typename std::tuple_element<N - 1, T>::type;

    static std::string get(const T& t) {
        return PrettyPrinter<TypeElem>::get(std::get<N - 1>(t));
    }
};

template <typename... Types>
struct TuplePrettyPrintHelper<0, Types...>
{
    using T = std::tuple<Types...>;

    static std::string get(const T& t) {
        return "";
    }
};


template <typename... Types>
struct PrettyPrinter<std::tuple<Types...>>
{
    using T = std::tuple<Types...>;

    static std::string get(const T& t) {
        return TuplePrettyPrintHelper<std::tuple_size_v<T>, Types...>::get(t);
    }
};

#define INSTANTIATE_TESTS(prefix, test_suite_name, generator) \
    INSTANTIATE_TEST_SUITE_P(prefix, test_suite_name, generator, \
        [](const testing::TestParamInfo<test_suite_name :: ParamType>& info) { \
            std::string name = PrettyPrinter<test_suite_name :: ParamType>::get(info.param); \
            for (int i = 0; i < name.length(); ++i) \
                if (!std::isalnum(name[i])) name[i] = '_'; \
          return name; \
        });

#endif  // TESTS_UTILS_H_
