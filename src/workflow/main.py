"""Runnable CLI demo for the Code Grounding DSPy Workflow.

Executes the 4-stage pipeline against a representative high-stakes technical query.
"""

import sys
from common.constants import MODEL_NAME_GEMINI_3_5_FLASH
from common.utils import dspy_configure, get_lm_for_model_name
from workflow.flow import CodeGroundingWorkflow


SAMPLE_QUESTION = "Does deleting a bidder in campaign manager automatically pause active Google Ads campaigns in shopping ad automation?"
SAMPLE_CONTEXT = """
Urgent incident investigation: Stakeholders observed that deleting a bidder record from the database
did not immediately pause live bidding campaigns on Google Ads. Need verifiable code trace of the deletion event flow.
"""

SAMPLE_CODE_CONTEXT = """
// BidderDeactivationListener.java
@EventListener
public void onBidderDeleted(BidderDeleteEvent event) {
    Long bidderId = event.getBidderId();
    // Soft delete: sets active = false
    bidderRepository.markInactive(bidderId);
    // NOTICE: Google Ads API campaign pause is handled via async cron exporter every 15 minutes,
    // NOT via synchronous event listener!
    log.info("Bidder {} marked inactive. Sync queue updated.", bidderId);
}

// GoogleAdsCampaignSyncJob.java
@Scheduled(fixedRate = 900_000)
public void syncPausedCampaigns() {
    List<Bidder> inactiveBidders = bidderRepository.findAllByActiveFalseAndCampaignsPausedFalse();
    for (Bidder bidder : inactiveBidders) {
        googleAdsClient.pauseCampaignsForBidder(bidder.getExternalId());
        bidder.setCampaignsPaused(true);
    }
}
"""


def run_demo(model_name: str = MODEL_NAME_GEMINI_3_5_FLASH):
    """Execute the full 4-stage workflow and display intermediate results."""
    print("=" * 80)
    print("🚀 DSPy Code Grounding Workflow Demo")
    print("=" * 80)

    # 1. Configure LM
    try:
        lm = get_lm_for_model_name(model_name, reasoning_effort="disable")
        dspy_configure(lm)
        print(f"✅ Configured LM: {model_name}\n")
    except Exception as e:
        print(f"❌ Failed to configure model {model_name}: {e}")
        print("Please ensure GEMINI_API_KEY or VERTEXAI_* credentials are set.")
        sys.exit(1)

    # 2. Instantiate workflow
    workflow = CodeGroundingWorkflow(use_react=False)

    print(f"📥 Input Question:\n{SAMPLE_QUESTION.strip()}\n")
    print("⏳ Executing 4-Stage Code Grounding Pipeline...\n")

    # 3. Execute
    result = workflow(
        raw_question=SAMPLE_QUESTION,
        operational_context=SAMPLE_CONTEXT,
        code_context=SAMPLE_CODE_CONTEXT
    )

    # 4. Display results
    print("-" * 80)
    print("📋 STAGE 1: INQUIRY RECORD (Ingress)")
    print("-" * 80)
    print(f"ID: {result.inquiry.inquiry_id}")
    print(f"Severity: {result.inquiry.stakes_severity}")
    print(f"Target Systems: {result.inquiry.target_systems}")
    print(f"Operational Context: {result.inquiry.operational_context}")

    print("\n" + "-" * 80)
    print("🔍 STAGE 2: PRIMARY BASELINE DISCOVERY (Forward Trace)")
    print("-" * 80)
    print(f"Call Entry: {result.baseline.call_tree.entry_point}")
    print(f"Intermediate Services: {result.baseline.call_tree.intermediate_services}")
    print(f"Leaf Mutations: {result.baseline.call_tree.leaf_mutations}")
    print(f"Citations: {result.baseline.code_citations}")
    print(f"Summary: {result.baseline.baseline_summary}")
    if result.baseline.diagram_d2:
        print(f"\nD2 Diagram:\n{result.baseline.diagram_d2}")

    print("\n" + "-" * 80)
    print("⚡ STAGE 3: ADVERSARIAL CHALLENGE REVIEW (RFC Proposals)")
    print("-" * 80)
    for p in result.ledger.proposals:
        print(f"• [{p.proposal_id}] Category: {p.category} | Target: {p.target_symbol}")
        print(f"  Failing Scenario: {p.failing_scenario}")
        print(f"  Suggested Revision: {p.suggested_revision}")

    print("\n" + "-" * 80)
    print("⚖️ STAGE 4: OPERATIONAL RECONCILIATION (Calibrated Verdict)")
    print("-" * 80)
    print(f"Certainty Score: {result.verdict.certainty_score}/5")
    print(f"Adopted Proposals: {result.verdict.adopted_proposals}")
    print(f"Rejected Proposals: {result.verdict.rejected_proposals}")
    print(f"Residual Risks: {result.verdict.residual_risks}")
    print(f"\n📢 Stakeholder Verdict:\n{result.verdict.plain_language_verdict}")
    print(f"\n🎯 Action Matrix:\n{result.verdict.actionable_decision_matrix}")
    print("=" * 80)


def main():
    run_demo()


if __name__ == "__main__":
    main()
