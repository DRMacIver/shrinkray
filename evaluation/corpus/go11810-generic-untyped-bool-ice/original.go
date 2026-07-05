// Command healthwatch polls a set of service endpoints and reports
// which ones have converged to the "ready" state. Predicate results are
// generic over bool-like types so callers can plug in their own flag
// types with String() methods.
package main

import (
	"fmt"
	"sort"
	"strings"
)

type ServiceState struct {
	Name     string
	State    string
	Restarts int
}

// StrictFlag is a defined bool used by callers that want method sets on
// their readiness flags.
type StrictFlag bool

func (f StrictFlag) String() string {
	if f {
		return "ready"
	}
	return "not-ready"
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func parseState(line string) (ServiceState, error) {
	fields := splitCSV(line)
	if len(fields) < 2 {
		return ServiceState{}, fmt.Errorf("malformed state line: %q", line)
	}
	state := ServiceState{Name: fields[0], State: fields[1]}
	if len(fields) > 2 {
		if _, err := fmt.Sscanf(fields[2], "%d", &state.Restarts); err != nil {
			return ServiceState{}, fmt.Errorf("bad restart count in %q", line)
		}
	}
	return state, nil
}

// isReady reports readiness as any bool-like flag type.
func isReady[B ~bool](state string) B {
	var ready B = state == "ready"
	return ready
}

func summarize(states []ServiceState) string {
	names := make([]string, 0, len(states))
	for _, s := range states {
		if isReady[StrictFlag](s.State) {
			names = append(names, s.Name)
		}
	}
	sort.Strings(names)
	return strings.Join(names, ",")
}

func worstRestarts(states []ServiceState) int {
	worst := 0
	for _, s := range states {
		if s.Restarts > worst {
			worst = s.Restarts
		}
	}
	return worst
}

func main() {
	input := []string{
		"gateway, ready, 0",
		"scheduler, degraded, 3",
		"store, ready, 1",
	}
	states := make([]ServiceState, 0, len(input))
	for _, line := range input {
		s, err := parseState(line)
		if err != nil {
			fmt.Println("skipping:", err)
			continue
		}
		states = append(states, s)
	}
	fmt.Println("ready:", summarize(states))
	fmt.Println("worst restart count:", worstRestarts(states))
	fmt.Println("gateway flag:", isReady[StrictFlag]("ready"))
}
