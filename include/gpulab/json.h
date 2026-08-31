#pragma once

#include <string>
#include <string_view>

namespace gpulab {
// Minimal JSON string escaping for metadata. Not a general JSON parser.
inline std::string json_quote(std::string_view input) {
  constexpr char hex[] = "0123456789abcdef";
  std::string output = "\"";
  for (const unsigned char ch : input) {
    switch (ch) {
    case '"': output += "\\\""; break;
    case '\\': output += "\\\\"; break;
    case '\n': output += "\\n"; break;
    case '\r': output += "\\r"; break;
    case '\t': output += "\\t"; break;
    default:
      if (ch < 0x20) {
        output += "\\u00";
        output += hex[ch >> 4];
        output += hex[ch & 0x0f];
      } else {
        output += static_cast<char>(ch);
      }
    }
  }
  return output + '"';
}
} // namespace gpulab
