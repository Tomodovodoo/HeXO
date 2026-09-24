#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nnue {
constexpr int patterns=177147, channels=32;
constexpr int powers[]={1,3,9,27,81,243,729,2187,6561,19683,59049};
struct Model {
    std::vector<int16_t> table=std::vector<int16_t>(patterns*channels);
    std::array<float,32*68> value_w{};
    std::array<float,32> value_b{},value_out{};
    float value_bias=0;
    std::array<float,16*104> policy_w{};
    std::array<float,16> policy_b{},policy_out{};
    float policy_bias=0;
    const int16_t* row(int code) const {return table.data()+size_t(code)*channels;}
    float value(const std::array<float,68>& input) const {
        auto hidden=value_b;
        // Input-major weights vectorize independent neurons. Each neuron keeps
        // the original input accumulation order, without reassociation.
        for(int j=0;j<68;++j)
            for(int h=0;h<32;++h) hidden[h]+=value_w[j*32+h]*input[j];
        float out=value_bias;
        for(int h=0;h<32;++h) out+=std::max(0.0f,hidden[h])*value_out[h];
        return out;
    }
};
using Handle=std::shared_ptr<const Model>;
inline thread_local std::string error;
inline Handle load(const char* path) {
    if(!path) throw std::runtime_error("Missing model path");
    std::ifstream file(std::filesystem::path(reinterpret_cast<const char8_t*>(path)),std::ios::binary);
    if(!file) throw std::runtime_error("Cannot open NNUE model");
    auto read=[&](void* out,size_t n) {
        if(!file.read(static_cast<char*>(out),std::streamsize(n)))
            throw std::runtime_error("Truncated NNUE model");
    };
    char magic[8];read(magic,8);
    if(std::memcmp(magic,"HXNNUE1\0",8)) throw std::runtime_error("Invalid NNUE magic");
    uint32_t dims[9];read(dims,sizeof(dims));
    constexpr uint32_t expected[]={1,0x01020304,177147,32,64,4,4,32,16};
    if(std::memcmp(dims,expected,sizeof(dims))) throw std::runtime_error("Unsupported NNUE format or dimensions");
    float scales[2];read(scales,sizeof(scales));
    if(scales[0]!=256 || scales[1]!=6000) throw std::runtime_error("Unsupported NNUE scales");
    uint64_t bytes;read(&bytes,sizeof(bytes));
    if(bytes!=uint64_t(patterns)*32*2+3938*4) throw std::runtime_error("Invalid NNUE payload size");
    auto model=std::make_shared<Model>();
    read(model->table.data(),model->table.size()*sizeof(int16_t));
    auto floats=[&](float* out,size_t n) {
        read(out,n*sizeof(float));
        for(size_t i=0;i<n;++i) if(!std::isfinite(out[i]) || std::abs(out[i])>1000000)
            throw std::runtime_error("Invalid NNUE head weight");
    };
    floats(model->value_w.data(),model->value_w.size());
    floats(model->value_b.data(),32);floats(model->value_out.data(),32);floats(&model->value_bias,1);
    floats(model->policy_w.data(),model->policy_w.size());
    floats(model->policy_b.data(),16);floats(model->policy_out.data(),16);floats(&model->policy_bias,1);
    // The file remains output-major; only the immutable runtime layout changes.
    auto value_rows=model->value_w;
    for(int h=0;h<32;++h) for(int j=0;j<68;++j) model->value_w[j*32+h]=value_rows[h*68+j];
    auto policy_rows=model->policy_w;
    for(int h=0;h<16;++h) for(int j=0;j<104;++j) model->policy_w[j*16+h]=policy_rows[h*104+j];
    if(file.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("Trailing NNUE payload");
    for(int code=0;code<patterns;++code) {
        int reverse=0,swap=0,n=code;
        for(int k=0;k<11;++k) {int d=n%3;n/=3;reverse=reverse*3+d;swap+=(d?3-d:0)*powers[k];}
        const auto* row=model->row(code);
        for(int j=0;j<32;++j) {
            if(row[j]<-256 || row[j]>256 || row[j]!=model->row(reverse)[j] ||
               row[j]!=(j<16?-model->row(swap)[j]:model->row(swap)[j]))
                throw std::runtime_error("NNUE table violates bounds or symmetry");
        }
    }
    return model;
}
}
