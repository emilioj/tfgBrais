#pragma once
#include <vector>

struct ImageData
{
    std::vector<unsigned char> data; // Raw image data
    int width;                       // Image width
    int height;                      // Image height
    int channels;                    // Number of channels (3 for RGB, 4 for RGBA)

    bool isEmpty() const
    {
        return data.empty() || width == 0 || height == 0 || channels == 0;
    }
};