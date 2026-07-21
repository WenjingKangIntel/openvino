// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_plugin_cache.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "common_test_utils/file_utils.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
namespace test {
namespace utils {

ov::AnyMap global_plugin_config = {};
std::unordered_set<std::string> available_devices = {};
std::string target_device = "";
std::string target_plugin_name = "";

namespace {
void log_create_core_event(const std::string& msg) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    std::clog << "[ov_plugin_cache][create_core][tid=" << std::this_thread::get_id() << "] " << msg << std::endl;
}
}  // namespace

void register_plugin(ov::Core& ov_core) noexcept {
    if (!target_plugin_name.empty()) {
        ov_core.register_plugin(target_plugin_name, target_device);
    }
}

void register_template_plugin([[maybe_unused]] ov::Core& ov_core) noexcept {
#if !defined(ENABLE_TEMPLATE_REGISTRATION)
    auto plugin_path =
        ov::util::make_plugin_library_name(ov::test::utils::getExecutableDirectory(),
                                           std::string(ov::test::utils::TEMPLATE_LIB) + OV_BUILD_POSTFIX);
    if (!ov::util::file_exists(plugin_path)) {
        OPENVINO_THROW("Plugin: " + plugin_path + " does not exists!");
    }
    ov_core.register_plugin(plugin_path, ov::test::utils::DEVICE_TEMPLATE);
#endif
}

ov::Core create_core(const std::string& in_target_device) {
    log_create_core_event("Enter create_core, requested_device='" + in_target_device + "'");
    ov::Core ov_core;
    log_create_core_event("ov::Core instance created");

#if !defined(OPENVINO_STATIC_LIBRARY) && !defined(USE_STATIC_IE)
    log_create_core_event("Registering target plugin (if configured)");
    register_plugin(ov_core);
    log_create_core_event("Target plugin registration step finished");
    // Register Template plugin as a reference provider
    log_create_core_event("Registering template plugin (if enabled)");
    register_template_plugin(ov_core);
    log_create_core_event("Template plugin registration step finished");
#endif  // !OPENVINO_STATIC_LIBRARY && !USE_STATIC_IE

    if (available_devices.empty()) {
        log_create_core_event("available_devices cache is empty, querying core devices");
        const auto core_devices = ov_core.get_available_devices();
        available_devices.insert(core_devices.begin(), core_devices.end());
        log_create_core_event("available_devices cache populated with " + std::to_string(available_devices.size()) + " entries");
    } else {
        log_create_core_event("available_devices cache already populated with " + std::to_string(available_devices.size()) + " entries");
    }

    if (!available_devices.count(in_target_device) && !in_target_device.empty()) {
        log_create_core_event("Requested device is not present in available_devices cache");
#ifndef NDEBUG
        std::cout << "Available devices :" << std::endl;
        for (const auto& device : available_devices) {
            std::cout << "    " << device << std::endl;
        }
#endif
        OPENVINO_THROW("No available devices for " + in_target_device);
    }

    if (!global_plugin_config.empty()) {
        log_create_core_event("Applying global_plugin_config");
        // apply config to main device specified by user at launch or to special device specified when creating new сore
        auto config_device = in_target_device.empty() ? target_device : in_target_device;
        for (auto& property : global_plugin_config) {
            try {
                log_create_core_event("Setting property '" + property.first + "' on device '" + config_device + "'");
                ov_core.set_property(config_device, global_plugin_config);
            } catch (...) {
                log_create_core_event("Setting property failed for key '" + property.first + "'");
                OPENVINO_THROW("Property " + property.first +
                               ", which was tried to set in --config file, is not supported by " + target_device);
            }
        }
        log_create_core_event("global_plugin_config applied successfully");
    } else {
        log_create_core_event("global_plugin_config is empty, skipping property setup");
    }
    log_create_core_event("Leave create_core");
    return ov_core;
}

namespace {
class TestListener : public testing::EmptyTestEventListener {
public:
    void OnTestEnd(const testing::TestInfo& testInfo) override {
        if (auto testResult = testInfo.result()) {
            if (testResult->Failed()) {
                PluginCache::get().reset();
            }
        }
    }
};
}  // namespace

PluginCache& PluginCache::get() {
    static PluginCache instance;
    return instance;
}

std::shared_ptr<ov::Core> PluginCache::core(const std::string& target_device) {
    std::lock_guard<std::mutex> lock(g_mtx);
    if (disable_plugin_cache) {
        return std::make_shared<ov::Core>(create_core(target_device));
    }
    if (!ov_core) {
        ov_core = std::make_shared<ov::Core>(create_core(target_device));
        assert(0 != ov_core.use_count());
    }
    return ov_core;
}

void PluginCache::reset() {
    std::lock_guard<std::mutex> lock(g_mtx);
    ov_core.reset();
}

PluginCache::PluginCache() {
    auto& listeners = testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TestListener);
    disable_plugin_cache = std::getenv("DISABLE_PLUGIN_CACHE") == nullptr ? false : true;
}
}  // namespace utils
}  // namespace test
}  // namespace ov
