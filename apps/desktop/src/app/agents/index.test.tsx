// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest'

import { group } from '@/components/pane-shell/tree/model'
import { $activeTreeGroup, $layoutTree } from '@/components/pane-shell/tree/store'
import { I18nProvider } from '@/i18n'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $sessionTiles } from '@/store/session-states'
import { $subagentsBySession, upsertSubagent } from '@/store/subagents'

import { AgentsView } from './index'

beforeAll(() => {
  Element.prototype.animate ??= (() => ({}) as Animation) as typeof Element.prototype.animate
})

function renderAgents() {
  return render(
    <I18nProvider configClient={null}>
      <AgentsView onClose={() => undefined} />
    </I18nProvider>
  )
}

describe('AgentsView session scope', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $activeTreeGroup.set(null)
    $layoutTree.set(null)
    $sessionTiles.set([])
    $subagentsBySession.set({})
  })

  afterEach(() => {
    cleanup()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $activeTreeGroup.set(null)
    $layoutTree.set(null)
    $sessionTiles.set([])
    $subagentsBySession.set({})
  })

  it('shows only current-session rows and keeps their counts accurate', () => {
    $activeSessionId.set('current')
    upsertSubagent('current', {
      goal: 'Current active task',
      status: 'running',
      subagent_id: 'current-active',
      task_index: 0
    })
    upsertSubagent('current', {
      goal: 'Current completed task',
      status: 'completed',
      subagent_id: 'current-done',
      task_index: 1
    })
    upsertSubagent('foreign', {
      goal: 'Foreign active task',
      status: 'running',
      subagent_id: 'foreign-active',
      task_index: 0
    })
    upsertSubagent('foreign', {
      goal: 'Foreign completed task',
      status: 'completed',
      subagent_id: 'foreign-done',
      task_index: 1
    })

    renderAgents()

    expect(screen.getByText('Current active task')).toBeTruthy()
    expect(screen.getByText('Current completed task')).toBeTruthy()
    expect(screen.queryByText('Foreign active task')).toBeNull()
    expect(screen.queryByText('Foreign completed task')).toBeNull()
    expect(screen.getByText('2 agents · 1 active')).toBeTruthy()
  })

  it('is empty when there is no focused runtime session', () => {
    upsertSubagent('foreign', {
      goal: 'Foreign active task',
      status: 'running',
      subagent_id: 'foreign-active',
      task_index: 0
    })

    renderAgents()

    expect(screen.getByText('No live subagents')).toBeTruthy()
    expect(screen.queryByText('Foreign active task')).toBeNull()
  })

  it('follows a focused tiled session instead of the primary runtime', () => {
    $activeSessionId.set('primary-runtime')
    $selectedStoredSessionId.set('primary-stored')
    $sessionTiles.set([{ storedSessionId: 'tiled-stored', runtimeId: 'tiled-runtime' }])
    $layoutTree.set(
      group(['workspace', 'session-tile:tiled-stored'], {
        id: 'focused-group',
        active: 'session-tile:tiled-stored'
      })
    )
    $activeTreeGroup.set('focused-group')
    upsertSubagent('primary-runtime', {
      goal: 'Primary foreign task',
      status: 'running',
      subagent_id: 'primary-agent',
      task_index: 0
    })
    upsertSubagent('tiled-runtime', {
      goal: 'Focused tiled task',
      status: 'running',
      subagent_id: 'tiled-agent',
      task_index: 0
    })

    renderAgents()

    expect(screen.getByText('Focused tiled task')).toBeTruthy()
    expect(screen.queryByText('Primary foreign task')).toBeNull()
    expect(screen.getByText('1 agent · 1 active')).toBeTruthy()
  })
})
