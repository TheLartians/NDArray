#pragma once

namespace lars {
  
  template <typename D,typename T = void> struct DummyTemplate{ const static bool value = true; using type = T; };
  
}