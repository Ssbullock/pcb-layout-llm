#!/usr/bin/env python3
"""
test_pcb_agent_harness.py
-------------------------
End-to-end integration tests for `pcb_agent_harness.py`:
  1. OpenAI / Qwen-Code / DeepSeek tool schema & XML fallback validation.
  2. Monster-board (`hackrf_pro` from `set1monster`, 581 parts) compact overview &
     localized `inspect_subsystem` token budget (< 4K tokens vs > 25K single-shot).
  3. Real Model-First failure catching in `validate_groups` on:
       - `AI_Hydro_Controller/inputs/groups.json` (duplicate C18/C22, missing D5..D9)
       - `robocup_kicker/inputs/groups.json` (dangling `near` ref `grp_pwm_md_1`, missing movable parts)
     plus repair verification and `validate_floorplan` (`kind="floorplan-intent"`, `decap_pin_distance`).
  4. Full closed-loop OpenAI tool-calling agent session on `ch32v203_ev`:
       overview -> subsystem -> validate -> place -> ordered subset route
       (GND planes -> +3.3V/+5V power rails -> `*` signals) -> inspect_routing_failures
       -> ripup_and_reroute (verifying trace preservation + clean DRC) ->
       patch_layout -> undo_last_patch -> run_routing_chain.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

WORKSPACE = Path("/Users/simon/Fine Tune")
CORPUS_DIR = WORKSPACE / "corpus"
MF_ROLLOUTS = WORKSPACE / "runs" / "frontier_benchmark" / "model_first_rollouts"
HARNESS_RUNS = WORKSPACE / "runs" / "harness_prototype"

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from pcb_agent_harness import (  # noqa: E402
    PCBAgentSession,
    TOOLS_SCHEMA,
    run_agent_loop,
)


def _fresh_dir(name: str) -> Path:
    d = HARNESS_RUNS / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestPCBAgentHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        HARNESS_RUNS.mkdir(parents=True, exist_ok=True)

    def test_01_tools_schema_openai_format(self) -> None:
        names = [t["function"]["name"] for t in TOOLS_SCHEMA]
        expected = {
            "get_board_overview",
            "inspect_subsystem",
            "validate_groups",
            "validate_floorplan",
            "run_placement",
            "route_nets",
            "run_routing_chain",
            "inspect_routing_failures",
            "ripup_and_reroute",
            "patch_layout",
            "undo_last_patch",
        }
        self.assertEqual(set(names), expected)
        for tool in TOOLS_SCHEMA:
            self.assertEqual(tool["type"], "function")
            fn = tool["function"]
            self.assertIsInstance(fn["description"], str)
            self.assertTrue(len(fn["description"]) > 10)
            self.assertEqual(fn["parameters"]["type"], "object")
            self.assertIn("properties", fn["parameters"])

    def test_02_monster_board_compact_overview_and_subsystem(self) -> None:
        work = _fresh_dir("test_monster_hackrf_pro")
        session = PCBAgentSession("hackrf_pro", CORPUS_DIR, work, set_name="set1monster")

        ov = session.get_board_overview()
        self.assertTrue(ov["ok"])
        self.assertGreater(ov["counts"]["total_parts"], 400)
        self.assertEqual(ov["layer_count"], 6)
        self.assertLess(
            ov["context_budget"]["est_tokens"],
            4000,
            f"Overview exceeded compact token target: {ov['context_budget']}",
        )
        self.assertTrue(ov["top_ics"], "Expected top_ics on hackrf_pro")

        # Pick the highest-pin IC and inspect its 1-hop neighborhood
        top_ic = ov["top_ics"][0]["ref"]
        sub = session.inspect_subsystem(top_ic, radius_hops=1, max_parts=40)
        self.assertTrue(sub["ok"])
        self.assertEqual(sub["resolved_as"], "part")
        self.assertLessEqual(len(sub["neighborhood_parts"]), 40)
        self.assertLess(
            sub["est_tokens"],
            5000,
            f"Subsystem exceeded compact token target: {sub['est_tokens']} est tokens",
        )

        # Inspect by power/signal net name too
        rail_net = ov["power_and_gnd_nets"]["rail_candidates"][0]["net"]
        sub_net = session.inspect_subsystem(rail_net, radius_hops=1, max_parts=25)
        self.assertTrue(sub_net["ok"])
        self.assertEqual(sub_net["resolved_as"], "net")
        self.assertLessEqual(len(sub_net["neighborhood_parts"]), 25)
        self.assertLess(sub_net["est_tokens"], 5000)

        # Non-existent query returns clean suggestions without crashing
        miss = session.inspect_subsystem("NONEXISTENT_XYZ_999")
        self.assertFalse(miss["ok"])
        self.assertIn("part_suggestions", miss)

        print(
            f"[PASS test_02] hackrf_pro ({ov['counts']['total_parts']} parts): "
            f"overview={ov['context_budget']['est_tokens']} tok, "
            f"subsystem({top_ic})={sub['est_tokens']} tok ({len(sub['neighborhood_parts'])} parts)"
        )

    def test_03_validate_groups_catches_real_model_first_failures(self) -> None:
        # 1) AI_Hydro_Controller: duplicate C18/C22 across grp_u10 & grp_u6, missing D5..D9
        hydro_work = _fresh_dir("test_validate_ai_hydro")
        hydro_sess = PCBAgentSession("AI_Hydro_Controller", CORPUS_DIR, hydro_work, set_name="sethydro")
        bad_hydro = json.loads((MF_ROLLOUTS / "AI_Hydro_Controller" / "inputs" / "groups.json").read_text())

        res_hydro = hydro_sess.validate_groups(bad_hydro)
        self.assertFalse(res_hydro["valid"])
        self.assertIn("C18", res_hydro["duplicate_refs"])
        self.assertIn("C22", res_hydro["duplicate_refs"])
        for missing_ref in ("D5", "D6", "D7", "D8", "D9"):
            self.assertIn(missing_ref, res_hydro["missing_movable_parts"])

        # Fix AI_Hydro_Controller groups (remove duplicate C18/C22 from grp_u6; add D5..D9 to grp_u9)
        fixed_hydro = json.loads(json.dumps(bad_hydro))
        for g in fixed_hydro["groups"]:
            if g["name"] == "grp_u6":
                g["refs"] = [r for r in g["refs"] if r not in ("C18", "C22")]
            if g["name"] == "grp_u9":
                for dref in ("D5", "D6", "D7", "D8", "D9"):
                    if dref not in g["refs"]:
                        g["refs"].append(dref)
        res_hydro_ok = hydro_sess.validate_groups(fixed_hydro)
        self.assertTrue(res_hydro_ok["valid"], f"Expected repaired hydro groups valid, got: {res_hydro_ok}")

        # 2) robocup_kicker: dangling `near` ref 'grp_pwm_md_1' across multiple groups
        rk_work = _fresh_dir("test_validate_robocup_kicker")
        rk_sess = PCBAgentSession("robocup_kicker", CORPUS_DIR, rk_work, set_name="set14")
        bad_rk = json.loads((MF_ROLLOUTS / "robocup_kicker" / "inputs" / "groups.json").read_text())

        res_rk = rk_sess.validate_groups(bad_rk)
        self.assertFalse(res_rk["valid"])
        dangling_targets = {d["target"] for d in res_rk["dangling_refs"]}
        self.assertIn("grp_pwm_md_1", dangling_targets)
        self.assertEqual(len(res_rk["dangling_refs"]), 4)
        self.assertTrue(any("grp_pwm_md_1" in e for e in res_rk["errors"]))

        # 3) validate_floorplan: ensures kind='floorplan-intent', catches bad kind & tight decap_pin_distance
        ch6_work = _fresh_dir("test_validate_floorplan_ch32v006")
        ch6_sess = PCBAgentSession("ch32v006_dev", CORPUS_DIR, ch6_work, set_name="set6")
        raw_intent = json.loads((MF_ROLLOUTS / "ch32v006_dev" / "inputs" / "intent.json").read_text())
        self.assertNotIn("kind", raw_intent)

        fp_res = ch6_sess.validate_floorplan(raw_intent)
        self.assertTrue(fp_res["valid"], f"Expected valid after kind normalization, got: {fp_res}")
        self.assertEqual(fp_res["kind"], "floorplan-intent")
        self.assertIn("kind='floorplan-intent'", fp_res["normalized_fields"])
        self.assertTrue(
            any("max_pin_distance_mm" in w for w in fp_res["warnings"]),
            f"Expected tight max_pin_distance_mm warning, got: {fp_res['warnings']}",
        )

        # Reject invalid kind
        bad_kind = dict(raw_intent, kind="not-a-floorplan")
        fp_bad = ch6_sess.validate_floorplan(bad_kind)
        self.assertFalse(fp_bad["valid"])

        print(
            f"[PASS test_03] AI_Hydro caught dups={list(res_hydro['duplicate_refs'])}, "
            f"missing={res_hydro['missing_movable_parts']}; "
            f"robocup_kicker caught dangling={sorted(dangling_targets)}"
        )

    def test_04_closed_loop_placement_ordered_routing_ripup_and_agent_loop(self) -> None:
        work = _fresh_dir("test_closed_loop_ch32v203_ev")
        session = PCBAgentSession("ch32v203_ev", CORPUS_DIR, work, set_name="set6")

        groups_doc = json.loads((MF_ROLLOUTS / "ch32v203_ev" / "inputs" / "groups.json").read_text())
        intent_doc = json.loads((MF_ROLLOUTS / "ch32v203_ev" / "inputs" / "intent.json").read_text())

        # Deterministic Qwen-Code / OpenAI tool-calling driver exercising the closed loop
        script_state: Dict[str, Any] = {"step": 0, "reroute_target": None}

        def scripted_qwen_openai_chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
            step = script_state["step"]
            script_state["step"] += 1

            # Turn 0: Explore overview + focal subsystem
            if step == 0:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_1", "type": "function", "function": {"name": "get_board_overview", "arguments": "{}"}},
                        {
                            "id": "tc_2",
                            "type": "function",
                            "function": {"name": "inspect_subsystem", "arguments": json.dumps({"ic_ref_or_net": "U1", "radius_hops": 1})},
                        },
                    ],
                }

            # Turn 1: Validate groups & floorplan (demonstrating Qwen XML fallback on one call too? Use standard OpenAI JSON!)
            if step == 1:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_3", "type": "function", "function": {"name": "validate_groups", "arguments": json.dumps({"groups_json": groups_doc})}},
                        {"id": "tc_4", "type": "function", "function": {"name": "validate_floorplan", "arguments": json.dumps({"intent_json": intent_doc})}},
                    ],
                }

            # Turn 2: Run placement
            if step == 2:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_5",
                            "type": "function",
                            "function": {
                                "name": "run_placement",
                                "arguments": json.dumps({"scattered": 2, "clustered": 2, "anchored": 0, "jobs": 2, "judge_count": 2}),
                            },
                        }
                    ],
                }

            # Turn 3: Ordered subset routing step 1 — GND inner planes
            if step == 3:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_6",
                            "type": "function",
                            "function": {
                                "name": "route_nets",
                                "arguments": json.dumps(
                                    {
                                        "nets": ["GND", "GND"],
                                        "plane_layers": ["In1.Cu", "In2.Cu"],
                                        "clearance": 0.1,
                                        "clearance_ceiling": 0.1,
                                        "via_size": 0.45,
                                        "via_drill": 0.2,
                                        "same_net_pad_clearance": 0.15,
                                    }
                                ),
                            },
                        }
                    ],
                }

            # Turn 4: Ordered subset routing step 2 — +3.3V & +5V power rails (0.4mm trunk + pad neckdown)
            if step == 4:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_7",
                            "type": "function",
                            "function": {
                                "name": "route_nets",
                                "arguments": json.dumps(
                                    {
                                        "nets": ["+3.3V", "+5V"],
                                        "track_width": 0.25,
                                        "power_nets": ["+3.3V", "+5V"],
                                        "power_nets_widths": [0.4, 0.4],
                                        "clearance": 0.1,
                                        "clearance_ceiling": 0.1,
                                        "via_size": 0.45,
                                        "via_drill": 0.2,
                                        "same_net_pad_clearance": 0.15,
                                        "via_cost": 110,
                                        "turn_cost": 1200,
                                    }
                                ),
                            },
                        }
                    ],
                }

            # Turn 5: Ordered subset routing step 3 — all remaining signal nets `*` + zone fill
            if step == 5:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_8",
                            "type": "function",
                            "function": {
                                "name": "route_nets",
                                "arguments": json.dumps(
                                    {
                                        "nets": ["*"],
                                        "track_width": 0.15,
                                        "clearance": 0.1,
                                        "clearance_ceiling": 0.1,
                                        "via_size": 0.45,
                                        "via_drill": 0.2,
                                        "same_net_pad_clearance": 0.15,
                                        "via_cost": 110,
                                        "turn_cost": 1200,
                                        "direction_preference_cost": 250,
                                        "bus": True,
                                        "ordering": "bus",
                                        "power_nets": ["+3.3V", "+5V", "GND"],
                                        "power_nets_widths": [0.5, 0.5, 0.5],
                                        "write_fill": True,
                                    }
                                ),
                            },
                        }
                    ],
                }

            # Turn 6: Inspect routing failures, DRC, and Teel sub-metrics
            if step == 6:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_9", "type": "function", "function": {"name": "inspect_routing_failures", "arguments": "{}"}}
                    ],
                }

            # Turn 7: Rescue any failing net OR ripup_and_reroute a routed signal net cleanly
            if step == 7:
                last_tool_payload = json.loads(messages[-1]["content"])
                failing_nets = list((last_tool_payload.get("blocking_by") or {}).get("nets") or [])
                rev = last_tool_payload.get("route_review") or {}
                cands = [x["net"] for x in (rev.get("longest_detours") or [])] + [x["net"] for x in (rev.get("most_vias") or [])]
                if failing_nets:
                    target_nets = failing_nets[:4]
                    script_state["reroute_target"] = target_nets[0]
                    script_state["had_failing"] = True
                    new_params = {
                        "track_width": 0.15,
                        "clearance": 0.1,
                        "clearance_ceiling": 0.1,
                        "via_size": 0.45,
                        "via_drill": 0.2,
                        "same_net_pad_clearance": 0.15,
                        "max_ripup": 12,
                        "rip_existing_nets": ["*"],
                        "write_fill": True,
                    }
                else:
                    target_net = cands[0] if cands else "/PA0"
                    target_nets = [target_net]
                    script_state["reroute_target"] = target_net
                    script_state["had_failing"] = False
                    new_params = {
                        "track_width": 0.18,
                        "clearance": 0.1,
                        "clearance_ceiling": 0.1,
                        "via_size": 0.45,
                        "via_drill": 0.2,
                        "same_net_pad_clearance": 0.15,
                        "write_fill": True,
                    }
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_10",
                            "type": "function",
                            "function": {
                                "name": "ripup_and_reroute",
                                "arguments": json.dumps(
                                    {
                                        "nets": target_nets,
                                        "new_params": new_params,
                                    }
                                ),
                            },
                        }
                    ],
                }

            # Turn 8: Re-inspect after ripup_and_reroute
            if step == 8:
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_11", "type": "function", "function": {"name": "inspect_routing_failures", "arguments": "{}"}}
                    ],
                }

            # Final message (Qwen `<tool_call>` not used -> completes loop)
            return {
                "role": "assistant",
                "content": f"Completed placement, ordered routing, DRC verification, and ripup_and_reroute on {script_state['reroute_target']}.",
            }

        transcript_file = work / "transcript.json"
        loop_res = run_agent_loop(
            session,
            scripted_qwen_openai_chat,
            max_turns=15,
            transcript_path=transcript_file,
        )
        self.assertTrue(loop_res["ok"])
        self.assertTrue(transcript_file.exists())
        self.assertEqual(loop_res["tool_calls_executed"], 11)

        # Pull tool outputs from messages for thorough assertions
        tool_outputs: Dict[str, List[Dict[str, Any]]] = {}
        for m in loop_res["messages"]:
            if m.get("role") == "tool":
                tool_outputs.setdefault(m["name"], []).append(json.loads(m["content"]))

        # Verify validation & placement
        self.assertTrue(tool_outputs["validate_groups"][0]["valid"])
        self.assertTrue(tool_outputs["validate_floorplan"][0]["valid"])
        place_res = tool_outputs["run_placement"][0]
        self.assertTrue(place_res["ok"])
        self.assertEqual(place_res["floorplan_errors"], 0)
        self.assertTrue(place_res["assembly_status"]["buildable"])

        # Verify ordered routing preserved non-conflicting pre-existing power rails during `*` signal route
        r_planes, r_pwr, r_sig = tool_outputs["route_nets"]
        self.assertTrue(r_planes["ok"])
        self.assertTrue(r_pwr["ok"])
        self.assertTrue(r_sig["ok"])
        self.assertGreaterEqual(r_sig["preserved_preexisting_nets_count"], 1)
        mod_names = {m["net"] for m in r_sig["modified_preexisting_nets"]}
        self.assertNotIn("+5V", mod_names)

        # Verify inspect_routing_failures before and after ripup_and_reroute
        insp_before, insp_after = tool_outputs["inspect_routing_failures"]
        self.assertGreater(insp_before["teel_score_100"], 50.0)

        rip_res = tool_outputs["ripup_and_reroute"][0]
        self.assertTrue(rip_res["ok"])
        self.assertTrue(rip_res["preserved_other_nets_after_undo"])
        if not script_state.get("had_failing"):
            self.assertGreater(rip_res["removed_segments"], 0)
            self.assertTrue(rip_res["preserved_other_nets_after_reroute"])

        self.assertEqual(insp_after["blocking"], 0, f"Post-ripup blocking != 0: {insp_after}")
        self.assertEqual(insp_after["kicad_drc"]["drc_errors"], 0)
        self.assertEqual(insp_after["kicad_drc"]["drc_unconnected"], 0)

        # Also test strict isolated ripup_and_reroute (no collateral rip) + rip-only + undo_last_patch!
        rev_after = insp_after.get("route_review") or {}
        sig_cands = [x["net"] for x in (rev_after.get("longest_detours") or [])] + [x["net"] for x in (rev_after.get("most_vias") or [])]
        target_net = sig_cands[0] if sig_cands else script_state["reroute_target"]

        iso_rip = session.ripup_and_reroute(
            [target_net],
            new_params={
                "track_width": 0.18,
                "clearance": 0.1,
                "clearance_ceiling": 0.1,
                "via_size": 0.45,
                "via_drill": 0.2,
                "same_net_pad_clearance": 0.15,
                "write_fill": True,
            },
        )
        self.assertTrue(iso_rip["ok"], f"Isolated reroute failed: {iso_rip}")
        self.assertGreater(iso_rip["removed_segments"], 0)
        self.assertTrue(iso_rip["preserved_other_nets_after_undo"])
        self.assertTrue(iso_rip["preserved_other_nets_after_reroute"])

        rip_only = session.ripup_and_reroute([target_net], reroute=False)
        self.assertTrue(rip_only["ok"])
        self.assertFalse(rip_only["rerouted"])
        self.assertTrue(rip_only["preserved_other_nets_after_undo"])
        self.assertGreater(rip_only["removed_segments"], 0)
        insp_unrouted = session.inspect_routing_failures()
        self.assertIn(target_net, insp_unrouted["blocking_by"]["nets"])
        # Confirm pad coordinates & bbox of failing net are populated!
        failing_entry = next(f for f in insp_unrouted["failing_nets_detail"] if f["net"] == target_net)
        self.assertIsNotNone(failing_entry["bbox_mm"])
        self.assertGreaterEqual(len(failing_entry["pads"]), 2)

        undo_res = session.undo_last_patch()
        self.assertTrue(undo_res["ok"])
        insp_restored = session.inspect_routing_failures()
        self.assertEqual(insp_restored["blocking"], 0)
        self.assertEqual(insp_restored["kicad_drc"]["drc_errors"], 0)

        # Also test patch_layout (intent_edits + rotation on placed board) & undo_last_patch
        patch_res = session.patch_layout(
            rotations={"C1": {"rot": 180, "relative": True}},
            preserve_unrelated_copper=True,
        )
        self.assertTrue(patch_res["ok"], f"patch_layout failed: {patch_res}")
        undo_patch = session.undo_last_patch()
        self.assertTrue(undo_patch["ok"])

        print(
            f"[PASS test_04] ch32v203_ev closed loop: blocking={insp_after['blocking']}, "
            f"drc_errors={insp_after['kicad_drc']['drc_errors']}, "
            f"teel={insp_after['teel_score_100']}, ripup_target={target_net}"
        )

    def test_05_qwen_xml_tool_call_and_routing_chain(self) -> None:
        work = _fresh_dir("test_xml_and_chain_ch32v203_ev")
        session = PCBAgentSession("ch32v203_ev", CORPUS_DIR, work, set_name="set6")
        groups_doc = json.loads((MF_ROLLOUTS / "ch32v203_ev" / "inputs" / "groups.json").read_text())
        intent_doc = json.loads((MF_ROLLOUTS / "ch32v203_ev" / "inputs" / "intent.json").read_text())
        strat_doc = json.loads((MF_ROLLOUTS / "ch32v203_ev" / "inputs" / "routing_strategy.json").read_text())

        turn = {"i": 0}

        def qwen_xml_chat(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
            i = turn["i"]
            turn["i"] += 1
            if i == 0:
                payload = json.dumps({"name": "validate_groups", "arguments": {"groups_json": groups_doc}})
                return {"role": "assistant", "content": f"<tool_call>\n{payload}\n</tool_call>"}
            if i == 1:
                payload = json.dumps({"name": "validate_floorplan", "arguments": {"intent_json": intent_doc}})
                return {"role": "assistant", "content": f"<tool_call>\n{payload}\n</tool_call>"}
            if i == 2:
                payload = json.dumps({"name": "run_placement", "arguments": {"scattered": 1, "clustered": 1, "anchored": 0, "jobs": 2, "judge_count": 1}})
                return {"role": "assistant", "content": f"<tool_call>\n{payload}\n</tool_call>"}
            if i == 3:
                payload = json.dumps({"name": "run_routing_chain", "arguments": {"strategy_json": strat_doc}})
                return {"role": "assistant", "content": f"<tool_call>\n{payload}\n</tool_call>"}
            return {"role": "assistant", "content": "Done via Qwen XML tool calls and run_routing_chain."}

        res = run_agent_loop(session, qwen_xml_chat, max_turns=8, transcript_path=work / "xml_transcript.json")
        self.assertTrue(res["ok"])
        self.assertEqual(res["tool_calls_executed"], 4)
        insp = session.inspect_routing_failures()
        self.assertTrue(insp["ok"])
        self.assertTrue(insp["kicad_drc"]["ran"])
        print(
            f"[PASS test_05] Qwen XML + run_routing_chain: blocking={insp['blocking']}, "
            f"drc_errors={insp['kicad_drc']['drc_errors']}, teel={insp['teel_score_100']}"
        )

    def test_06_qwen3_function_xml_masking_and_graded_model_first_reward(self) -> None:
        from pcb_agent_harness import _mask_stale_observations, _parse_qwen_xml_tool_calls
        from gym_rollout import compute_rl_reward, _diagnose_raw_groups_invariants, parse_kicad_pcb

        # 1. Verify Qwen3-Coder <function=><parameter=> XML parsing
        xml_txt = (
            "<tool_call>\n"
            "<function=validate_floorplan>\n"
            '<parameter=intent_json>{"schema": 1, "kind": "floorplan-intent", "assembly": {"sides": "front"}}</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        parsed = _parse_qwen_xml_tool_calls(xml_txt)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["function"]["name"], "validate_floorplan")
        args_obj = json.loads(parsed[0]["function"]["arguments"])
        self.assertEqual(args_obj["intent_json"]["kind"], "floorplan-intent")

        # 2. Verify Complexity-Trap observation masking keeps recent tool outputs & masks old >1600 char dumps
        big_ok = json.dumps({"valid": True, "blob": "x" * 2000})
        big_recent = json.dumps({"valid": False, "errors": ["err"], "blob": "y" * 2000})
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "tool", "name": "get_board_overview", "content": big_ok},
            {"role": "tool", "name": "inspect_subsystem", "content": big_ok},
            {"role": "tool", "name": "validate_groups", "content": big_recent},
        ]
        masked = _mask_stale_observations(msgs, keep_recent_tool_msgs=1, max_obs_chars=1600)
        self.assertIn("masked_stale_observation", masked[1]["content"])
        self.assertIn("masked_stale_observation", masked[2]["content"])
        self.assertEqual(masked[3]["content"], big_recent)

        # 3. Verify _diagnose_raw_groups_invariants + graded compute_rl_reward handles bare list without crash
        pcb = parse_kicad_pcb(str(MF_ROLLOUTS / "robocup_kicker" / "fixed.kicad_pcb"))
        raw_list = json.loads((MF_ROLLOUTS / "robocup_kicker" / "inputs" / "groups.json").read_text())["groups"]
        norm_doc, diag, pen = _diagnose_raw_groups_invariants(raw_list, pcb, set())
        self.assertTrue(diag["non_object_wrapper"])
        self.assertGreater(pen, 25.0)
        graded_fail = compute_rl_reward({}, schema_invariant_penalty=pen)
        self.assertGreater(graded_fail, -500.0)
        print(f"[PASS test_06] Qwen3 XML + masking + graded failure reward={graded_fail} (diag={diag})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
