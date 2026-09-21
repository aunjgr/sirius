/* Copyright 2026 Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "catch.hpp"
#include "exec/scoped_dispatcher.hpp"

#include <latch>

TEST_CASE("scoped dispatcher try_schedule rejects work without retaining it", "[scoped_dispatcher]")
{
  sirius::exec::static_thread_pool pool(2, "dispatcher_test");
  sirius::exec::scoped_dispatcher dispatcher(pool, 1);
  std::latch started{1};
  std::latch release{1};
  std::latch unexpected{1};

  REQUIRE(dispatcher.try_schedule([&] {
    started.count_down();
    release.wait();
  }));
  started.wait();
  bool rejected = !dispatcher.try_schedule([&] { unexpected.count_down(); });
  release.count_down();
  dispatcher.wait_for_all();
  REQUIRE(rejected);
  CHECK_FALSE(unexpected.try_wait());

  std::latch accepted{1};
  CHECK(dispatcher.try_schedule([&] { accepted.count_down(); }));
  accepted.wait();
  dispatcher.wait_for_all();
  dispatcher.request_stop();
  CHECK_FALSE(dispatcher.try_schedule([] {}));
}
