/*
 * Copyright 2026 Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "offload/substrait_execution.hpp"
#include "substrait/plan.pb.h"

#include <catch.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace {

using sirius::offload::resolved_tae_read;
using sirius::offload::substrait_error_code;
using sirius::offload::substrait_execution_error;
using sirius::offload::tae_read;
using sirius::offload::tae_read_resolver;

void append_varint(std::string& output, std::uint64_t value)
{
  while (value >= 0x80U) {
    output.push_back(static_cast<char>((value & 0x7fU) | 0x80U));
    value >>= 7U;
  }
  output.push_back(static_cast<char>(value));
}

void append_varint_field(std::string& output, unsigned field, std::uint64_t value)
{
  append_varint(output, static_cast<std::uint64_t>(field) << 3U);
  append_varint(output, value);
}

void append_bytes_field(std::string& output, unsigned field, const std::string& value)
{
  append_varint(output, (static_cast<std::uint64_t>(field) << 3U) | 2U);
  append_varint(output, value.size());
  output.append(value);
}

std::string make_tae_read(std::uint64_t feature_bits = 0, std::uint64_t expires_at = 2000)
{
  std::string result;
  append_varint_field(result, 1, 1);
  if (feature_bits != 0) { append_varint_field(result, 2, feature_bits); }
  append_bytes_field(result, 3, "opaque-read-ref");
  append_bytes_field(result, 4, "query-1");
  append_varint_field(result, 5, 42);
  append_varint_field(result, 6, 84);
  append_bytes_field(result, 7, std::string(12, 's'));
  append_bytes_field(result, 8, std::string(32, 'd'));
  append_bytes_field(result, 9, std::string(32, 'm'));
  append_bytes_field(result, 10, std::string(32, 'c'));
  append_varint_field(result, 11, expires_at);
  return result;
}

::substrait::NamedStruct make_schema()
{
  ::substrait::NamedStruct schema;
  schema.add_names("value");
  schema.mutable_struct_()->add_types()->mutable_i64();
  return schema;
}

::substrait::Plan make_read_plan(const std::string& tae_read_bytes)
{
  ::substrait::Plan plan;
  plan.add_expected_type_urls(std::string(sirius::offload::k_tae_read_type_url));
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("value");
  auto* read = root->mutable_input()->mutable_read();
  read->mutable_base_schema()->CopyFrom(make_schema());
  auto* detail = read->mutable_extension_table()->mutable_detail();
  detail->set_type_url(std::string(sirius::offload::k_tae_read_type_url));
  detail->set_value(tae_read_bytes);
  return plan;
}

class fake_resolution final : public resolved_tae_read {
 public:
  fake_resolution(::substrait::NamedStruct schema,
                  bool capability_mismatch,
                  bool read_ref_mismatch,
                  int& destroyed)
    : schema_(std::move(schema)),
      capability_mismatch_(capability_mismatch),
      read_ref_mismatch_(read_ref_mismatch),
      destroyed_(destroyed)
  {
  }

  ~fake_resolution() noexcept override { ++destroyed_; }

  const std::string& relation_name() const noexcept override { return relation_name_; }
  const ::substrait::NamedStruct& canonical_schema() const noexcept override { return schema_; }
  const std::string& read_ref() const noexcept override
  { return read_ref_mismatch_ ? bad_read_ref_ : read_ref_; }
  const std::string& query_id() const noexcept override { return query_id_; }
  std::uint64_t account_id() const noexcept override { return 42; }
  std::uint64_t table_id() const noexcept override { return 84; }
  const std::string& snapshot_ts() const noexcept override { return snapshot_ts_; }
  const std::string& schema_digest() const noexcept override { return schema_digest_; }
  const std::string& manifest_sha256() const noexcept override { return manifest_sha256_; }
  const std::string& capability_hash() const noexcept override
  { return capability_mismatch_ ? bad_capability_hash_ : capability_hash_; }
  std::uint64_t expires_at_unix_ms() const noexcept override { return 2000; }

 private:
  ::substrait::NamedStruct schema_;
  bool capability_mismatch_;
  bool read_ref_mismatch_;
  int& destroyed_;
  std::string relation_name_       = "tae_query_1";
  std::string read_ref_            = "opaque-read-ref";
  std::string bad_read_ref_        = "different-read-ref";
  std::string query_id_            = "query-1";
  std::string snapshot_ts_         = std::string(12, 's');
  std::string schema_digest_       = std::string(32, 'd');
  std::string manifest_sha256_     = std::string(32, 'm');
  std::string capability_hash_     = std::string(32, 'c');
  std::string bad_capability_hash_ = std::string(32, 'x');
};

class fake_resolver final : public tae_read_resolver {
 public:
  std::unique_ptr<resolved_tae_read> resolve(const tae_read& request,
                                             const ::substrait::NamedStruct& schema) override
  {
    ++calls;
    last_read_ref = request.read_ref;
    return std::make_unique<fake_resolution>(schema, mismatch, read_ref_mismatch, destroyed);
  }

  bool mismatch          = false;
  bool read_ref_mismatch = false;
  int calls              = 0;
  int destroyed          = 0;
  std::string last_read_ref;
};

void require_error(const std::string& plan,
                   fake_resolver& resolver,
                   substrait_error_code expected,
                   bool fallback)
{
  try {
    (void)sirius::offload::detail::validate_and_resolve_substrait(plan, resolver, 1000);
    FAIL("expected Substrait validation to fail");
  } catch (const substrait_execution_error& error) {
    REQUIRE(error.code() == expected);
    REQUIRE(error.fallback_eligible() == fallback);
  }
}

}  // namespace

TEST_CASE("Substrait TaeRead is authenticated and rewritten without execution",
          "[substrait_contract]")
{
  fake_resolver resolver;
  auto input = make_read_plan(make_tae_read()).SerializeAsString();
  {
    auto validated = sirius::offload::detail::validate_and_resolve_substrait(input, resolver, 1000);
    REQUIRE(resolver.calls == 1);
    REQUIRE(resolver.last_read_ref == "opaque-read-ref");
    REQUIRE(resolver.destroyed == 0);
    REQUIRE(validated.resolutions.size() == 1);

    ::substrait::Plan rewritten;
    REQUIRE(rewritten.ParseFromString(validated.serialized));
    const auto& read = rewritten.relations(0).root().input().read();
    REQUIRE(read.has_named_table());
    REQUIRE(read.named_table().names_size() == 1);
    REQUIRE(read.named_table().names(0) == "tae_query_1");
    REQUIRE_FALSE(read.has_extension_table());
  }
  REQUIRE(resolver.destroyed == 1);
}

TEST_CASE("Substrait capability failures never reach GPU planning", "[substrait_contract]")
{
  SECTION("expired capability")
  {
    fake_resolver resolver;
    auto plan = make_read_plan(make_tae_read(0, 1000)).SerializeAsString();
    require_error(plan, resolver, substrait_error_code::AUTHENTICATION_FAILED, false);
    REQUIRE(resolver.calls == 0);
  }

  SECTION("resolver metadata mismatch")
  {
    fake_resolver resolver;
    resolver.mismatch = true;
    auto plan         = make_read_plan(make_tae_read()).SerializeAsString();
    require_error(plan, resolver, substrait_error_code::AUTHENTICATION_FAILED, false);
    REQUIRE(resolver.calls == 1);
    REQUIRE(resolver.destroyed == 1);
  }

  SECTION("resolver read_ref mismatch")
  {
    fake_resolver resolver;
    resolver.read_ref_mismatch = true;
    auto plan                  = make_read_plan(make_tae_read()).SerializeAsString();
    require_error(plan, resolver, substrait_error_code::AUTHENTICATION_FAILED, false);
    REQUIRE(resolver.calls == 1);
    REQUIRE(resolver.destroyed == 1);
  }
}

TEST_CASE("only explicit Sirius v1 capabilities are fallback eligible", "[substrait_contract]")
{
  SECTION("unknown feature bit")
  {
    fake_resolver resolver;
    auto plan = make_read_plan(make_tae_read(1)).SerializeAsString();
    require_error(plan, resolver, substrait_error_code::UNSUPPORTED_PLAN, true);
    REQUIRE(resolver.calls == 0);
  }

  SECTION("join relation")
  {
    fake_resolver resolver;
    ::substrait::Plan plan;
    plan.add_relations()->mutable_root()->mutable_input()->mutable_join();
    require_error(plan.SerializeAsString(), resolver, substrait_error_code::UNSUPPORTED_PLAN, true);
    REQUIRE(resolver.calls == 0);
  }
}

TEST_CASE("malformed and unknown Substrait input is rejected as invalid", "[substrait_contract]")
{
  SECTION("malformed bytes")
  {
    fake_resolver resolver;
    require_error("\x0a\xff", resolver, substrait_error_code::INVALID_PLAN, false);
  }

  SECTION("unknown plan field")
  {
    fake_resolver resolver;
    auto bytes = make_read_plan(make_tae_read()).SerializeAsString();
    append_varint(bytes, (99U << 3U));
    append_varint(bytes, 1);
    require_error(bytes, resolver, substrait_error_code::INVALID_PLAN, false);
    REQUIRE(resolver.calls == 0);
  }
}
