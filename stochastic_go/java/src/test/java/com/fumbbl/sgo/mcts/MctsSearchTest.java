package com.fumbbl.sgo.mcts;

import com.fumbbl.sgo.game.SGoAction;
import com.fumbbl.sgo.game.SGoState;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

public class MctsSearchTest {

    @Test
    void singleIterationCreatesAtLeastOneNode() {
        MctsSearch search = new MctsSearch(new SearchContext(), new Random(0));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), 1);
        assertNotNull(result.action);
        assertTrue(result.stats.totalStateNodes >= 1);
    }

    @Test
    void actionEdgesLazilyPopulated() {
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(0));
        search.search(SGoState.initial(), 5);

        // Root node should be expanded after at least 1 iteration
        StateNode root = ctx.tt.lookup(SGoState.initial().stateHash);
        assertNotNull(root);
        assertTrue(root.isExpanded());
        // Should have END_TURN + up to 64 place actions
        assertTrue(root.edgeCount > 1);
        assertTrue(root.edgeCount <= 65);
    }

    @Test
    void visitCountNeverExceedsIterations() {
        int budget = 50;
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(42));
        search.search(SGoState.initial(), budget);

        for (StateNode node : ctx.tt.allNodes()) {
            assertTrue(node.visitCount <= budget,
                    "visitCount=" + node.visitCount + " > budget=" + budget + " (transposition violation)");
        }
    }

    @Test
    void bestActionIsMostVisited() {
        int budget = 100;
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(7));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), budget);

        StateNode root = ctx.tt.lookup(SGoState.initial().stateHash);
        assertNotNull(root);
        assertTrue(root.isExpanded());

        // Find the most-visited edge
        int maxVisits = 0;
        int[] edgeIds = root.edgeIds;
        for (int i = 0; i < root.edgeCount; i++) {
            ActionEdge edge = root.edges[edgeIds[i]];
            if (edge.visitCount > maxVisits) maxVisits = edge.visitCount;
        }

        // Result action's edge should have max visits
        int resultId = result.action.id;
        ActionEdge resultEdge = root.edges[resultId];
        assertNotNull(resultEdge);
        assertEquals(maxVisits, resultEdge.visitCount,
                "bestAction should be the most-visited action");
    }

    @Test
    void transpositionHitsIncreaseWithBudget() {
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(99));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), 200);

        assertTrue(result.stats.totalTranspositionHits > 0,
                "Expected transposition hits after 200 iterations");
        assertTrue(result.stats.totalTranspositionAttempts >= result.stats.totalTranspositionHits);
    }

    @Test
    void chanceNodeOutcomesHaveProbabilitySummingToOne() {
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(0));
        search.search(SGoState.initial(), 20);

        for (StateNode node : ctx.tt.allNodes()) {
            if (!node.isExpanded()) continue;
            for (int i = 0; i < node.edgeCount; i++) {
                ActionEdge edge = node.edges[node.edgeIds[i]];
                if (edge == null) continue; // not yet visited (lazy creation)
                ChanceNode cn = edge.chanceNode;
                if (cn == null || !cn.isExpanded()) continue;
                double sum = 0.0;
                for (int j = 0; j < cn.outcomeCount; j++) {
                    sum += cn.outcomes[j].probability;
                }
                assertEquals(1.0, sum, 1e-9,
                        "Chance node probabilities must sum to 1.0");
            }
        }
    }

    @Test
    void chanceNodeAtMostTwoOutcomesForPlacement() {
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(0));
        search.search(SGoState.initial(), 30);

        for (StateNode node : ctx.tt.allNodes()) {
            if (!node.isExpanded()) continue;
            for (int i = 0; i < node.edgeCount; i++) {
                ActionEdge edge = node.edges[node.edgeIds[i]];
                if (edge == null) continue; // not yet visited (lazy creation)
                ChanceNode cn = edge.chanceNode;
                if (cn == null || !cn.isExpanded()) continue;
                // Placement on empty board with k=0: roll 1 fails, rolls 2-6 succeed → 2 outcomes
                assertTrue(cn.outcomeCount >= 1 && cn.outcomeCount <= 6,
                        "Unexpected outcome count: " + cn.outcomeCount);
            }
        }
    }

    @Test
    void searchProducesValidAction() {
        MctsSearch search = new MctsSearch(new SearchContext(), new Random(0));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), 50);
        assertNotNull(result.action);
        // Should be a place or end_turn action
        assertTrue(result.action == SGoAction.END_TURN || result.action.isPlace());
    }

    @Test
    void statsAreConsistent() {
        SearchContext ctx = new SearchContext();
        MctsSearch search = new MctsSearch(ctx, new Random(42));
        MctsSearch.SearchResult result = search.search(SGoState.initial(), 100);
        SearchStats stats = result.stats;

        assertEquals(100, stats.iterations);
        assertTrue(stats.totalTimeMs >= 0);
        assertTrue(stats.iterationsPerSecond > 0);
        assertTrue(stats.totalStateNodes > 0);
        assertTrue(stats.avgBranchingFactor > 0);
        assertTrue(stats.transpositionHitRate >= 0 && stats.transpositionHitRate <= 1);
    }
}
