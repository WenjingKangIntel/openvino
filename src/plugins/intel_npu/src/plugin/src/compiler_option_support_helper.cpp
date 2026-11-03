// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiler_option_support_helper.hpp"

#include <algorithm>
#include <memory>
#include <sstream>

#include "intel_npu/common/option_support_cache.hpp"

namespace intel_npu {

namespace {
OptionSupportCache::CacheKey toCacheKey(const ov::intel_npu::CompilerType compilerType) {
    return static_cast<OptionSupportCache::CacheKey>(compilerType);
}

std::string toString(const ov::intel_npu::CompilerType compilerType) {
    std::ostringstream stream;
    stream << compilerType;
    return stream.str();
}

std::string joinOptions(const std::vector<std::string>& options) {
    std::ostringstream stream;
    for (size_t index = 0; index < options.size(); ++index) {
        if (index != 0) {
            stream << " ";
        }
        stream << options[index];
    }
    return stream.str();
}
}  // namespace

CompilerOptionSupportHelper::CompilerOptionSupportHelper(const ov::SoPtr<IEngineBackend>& backend,
                                                         const CompilerAdapterFactory& adapterFactory)
    : _backend(backend),
      _adapterFactory(adapterFactory),
      _optionSupportCache(std::make_shared<OptionSupportCache>()),
      _logger("CompilerOptionSupportHelper", Logger::global().level()) {}

const std::shared_ptr<OptionSupportCache>& CompilerOptionSupportHelper::getOptionSupportCache() const {
    return _optionSupportCache;
}

bool CompilerOptionSupportHelper::isOptionSupported(ov::intel_npu::CompilerType compilerType,
                                                    const std::string& optionName,
                                                    const std::optional<std::string>& optionValue) {
    const auto supportedCompilerTypes = _adapterFactory.getKnownCompilerTypes();
    OPENVINO_ASSERT(std::find(supportedCompilerTypes.begin(), supportedCompilerTypes.end(), compilerType) !=
                        supportedCompilerTypes.end(),
                    "Unsupported compiler type");

    const auto cacheKey = toCacheKey(compilerType);
    const auto getCachedSupport = [&]() -> std::optional<bool> {
        if (optionValue.has_value()) {
            return std::nullopt;
        }
        return _optionSupportCache->isOptionSupported(cacheKey, optionName);
    };

    if (const auto cachedSupport = getCachedSupport(); cachedSupport.has_value()) {
        return cachedSupport.value();
    }

    std::unique_ptr<ICompilerAdapter> compiler;
    compiler = _adapterFactory.getCompiler(_backend, compilerType, "", _optionSupportCache);

    std::once_flag* optionsLoaded = nullptr;
    {
        std::lock_guard<std::mutex> lock(_supportedOptionsLoadedMutex);
        optionsLoaded = &_supportedOptionsLoaded[cacheKey];
    }
    std::call_once(*optionsLoaded, [&]() {
        const auto supportedOptions = compiler->get_supported_options();
        if (_logger.level() >= ov::log::Level::INFO) {
            _logger.info("Compiler %s reports %zu supported options: {%s}",
                         toString(compilerType).c_str(),
                         supportedOptions.size(),
                         joinOptions(supportedOptions).c_str());
        }
    });

    if (const auto cachedSupport = getCachedSupport(); cachedSupport.has_value()) {
        return cachedSupport.value();
    }

    return compiler->is_option_supported(optionName, optionValue);
}

}  // namespace intel_npu
