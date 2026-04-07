package com.fumbbl.ffb.server.injury.injuryType;

import com.fumbbl.ffb.ApothecaryMode;
import com.fumbbl.ffb.FactoryType;
import com.fumbbl.ffb.FieldCoordinate;
import com.fumbbl.ffb.PlayerState;
import com.fumbbl.ffb.SpecialEffect;
import com.fumbbl.ffb.factory.ArmorModifierFactory;
import com.fumbbl.ffb.factory.InjuryModifierFactory;
import com.fumbbl.ffb.injury.InjuryType;
import com.fumbbl.ffb.injury.context.InjuryContext;
import com.fumbbl.ffb.model.Game;
import com.fumbbl.ffb.model.Player;
import com.fumbbl.ffb.modifiers.SpecialEffectArmourModifier;
import com.fumbbl.ffb.server.DiceInterpreter;
import com.fumbbl.ffb.server.DiceRoller;
import com.fumbbl.ffb.server.GameState;
import com.fumbbl.ffb.server.step.IStep;

import java.util.Arrays;

public abstract class AbstractInjuryTypeBombWithModifier<T extends InjuryType> extends InjuryTypeServer<T> {

	AbstractInjuryTypeBombWithModifier(T injuryType) {
		super(injuryType);
	}

	@Override
	public void handleInjury(IStep step, Game game, GameState gameState, DiceRoller diceRoller,
	                         Player<?> pAttacker, Player<?> pDefender, FieldCoordinate pDefenderCoordinate, FieldCoordinate fromCoordinate, InjuryContext pOldInjuryContext,
	                         ApothecaryMode pApothecaryMode) {

		// in BB2020/2025 bombs place players prone, chainsaw only takes effect on falling down or being knocked down
		// hence chainsaw is ignored here
		DiceInterpreter diceInterpreter = DiceInterpreter.getInstance();
		injuryContext.setArmorRoll(diceRoller.rollArmour());
		injuryContext.setArmorBroken(diceInterpreter.isArmourBroken(gameState, injuryContext));

		if (!injuryContext.isArmorBroken()) {
			((ArmorModifierFactory) game.getFactory(FactoryType.Factory.ARMOUR_MODIFIER)).specialEffectArmourModifiers(SpecialEffect.BOMB, pDefender)
				.forEach(injuryContext::addArmorModifier);
				injuryContext.setArmorBroken(diceInterpreter.isArmourBroken(gameState, injuryContext));
		}

		if (injuryContext.isArmorBroken()) {
			injuryContext.setInjuryRoll(diceRoller.rollInjury());
			InjuryModifierFactory factory = game.getFactory(FactoryType.Factory.INJURY_MODIFIER);
			factory.findInjuryModifiers(game, injuryContext, injuryType.isCausedByOpponent() ? pAttacker : null, pDefender, isStab(), isFoul(), isVomitLike()).forEach(injuryContext::addInjuryModifier);

			if (Arrays.stream(injuryContext.getArmorModifiers())
				.noneMatch(modifier -> modifier instanceof SpecialEffectArmourModifier)) {
				factory.specialEffectInjuryModifiers(SpecialEffect.BOMB)
					.forEach(injuryContext::addInjuryModifier);
			}
			setInjury(pDefender, gameState, diceRoller);

		} else {
			injuryContext.setInjury(new PlayerState(PlayerState.PRONE));
		}
	}
}