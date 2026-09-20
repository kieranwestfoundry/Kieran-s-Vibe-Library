import nuke
import json

def create_llm_archetype_node():
    """Generates a NoOp node configured for LLM Archetype management."""
    node = nuke.createNode("NoOp", "name LLM_Archetype")
    
    # 1. UI Tab
    node.addKnob(nuke.Tab_Knob("llm_settings", "LLM Settings"))
    
    # 2. Hidden Data Storage (The "Database")
    default_archetypes = {
        "Default Assistant": {
            "system_prompt": "You are a helpful assistant for Nuke compositing.",
            "temperature": 0.7,
            "top_p": 1.0
        },
        "Senior Pipeline TD": {
            "system_prompt": "You are an expert Python VFX Pipeline TD. Write highly optimized, PEP-8 compliant code for Foundry Nuke.",
            "temperature": 0.1,
            "top_p": 0.95
        },
        "Creative Brainstormer": {
            "system_prompt": "You are a creative director helping brainstorm visual effects concepts.",
            "temperature": 0.9,
            "top_p": 1.0
        }
    }
    
    storage_knob = nuke.String_Knob("archetype_data", "Archetype Data")
    storage_knob.setValue(json.dumps(default_archetypes))
    storage_knob.setFlag(nuke.INVISIBLE)
    node.addKnob(storage_knob)
    
    # 3. The Dropdown
    archetype_names = list(default_archetypes.keys())
    dropdown_knob = nuke.Enumeration_Knob("archetype_select", "Archetype", archetype_names)
    node.addKnob(dropdown_knob)
    
    # 4. Model Parameters
    prompt_knob = nuke.Multiline_Eval_String_Knob("system_prompt", "System Prompt")
    node.addKnob(prompt_knob)
    
    temp_knob = nuke.Double_Knob("temperature", "Temperature")
    temp_knob.setRange(0.0, 2.0)
    node.addKnob(temp_knob)
    
    topp_knob = nuke.Double_Knob("top_p", "Top P")
    topp_knob.setRange(0.0, 1.0)
    node.addKnob(topp_knob)
    
    # 5. Add / Save Button
    node.addKnob(nuke.Text_Knob("divider", ""))
    save_btn = nuke.PyScript_Knob("save_archetype", "Save Current as New Archetype")
    node.addKnob(save_btn)

    # 6. Callbacks Setup
    # knobChanged: Triggered when the dropdown is changed
    knob_changed_code = """
import json
node = nuke.thisNode()
knob = nuke.thisKnob()

if knob.name() == "archetype_select":
    try:
        data = json.loads(node['archetype_data'].value())
        selected = knob.value()
        if selected in data:
            node['system_prompt'].setValue(data[selected].get('system_prompt', ''))
            node['temperature'].setValue(data[selected].get('temperature', 0.7))
            node['top_p'].setValue(data[selected].get('top_p', 1.0))
    except Exception as e:
        print(f"Error loading archetype: {e}")
"""
    node['knobChanged'].setValue(knob_changed_code)

    # PyScript Button Callback: Triggered when user clicks "Save"
    save_btn_code = """
import json
import nuke

node = nuke.thisNode()
new_name = nuke.getInput("Enter name for new Archetype:")

if new_name:
    try:
        # Get current data
        data = json.loads(node['archetype_data'].value())
        
        # Build new entry from current UI state
        data[new_name] = {
            "system_prompt": node['system_prompt'].value(),
            "temperature": node['temperature'].value(),
            "top_p": node['top_p'].value()
        }
        
        # Save back to hidden knob
        node['archetype_data'].setValue(json.dumps(data))
        
        # Update dropdown choices
        node['archetype_select'].setValues(list(data.keys()))
        node['archetype_select'].setValue(new_name)
        
    except Exception as e:
        nuke.message(f"Failed to save archetype: {e}")
"""
    save_btn.setValue(save_btn_code)

    # Force initial population of the knobs
    node['archetype_select'].setValue("Senior Pipeline TD")
    
    return node

# Execute
create_llm_archetype_node()