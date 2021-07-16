-- path to save rendering ouputs
root_path = "E:/Gits/Datasets/DCN/scene01"
main_path = root_path + "/picMain/"
aux_path = root_path + "/picAux/"
depth_path = root_path + "/depth/"
makeDir main_path all:true
makeDir aux_path all:true
makeDir depth_path all:true

-- set up rendering sequence
N = 16 -- how many frames to render
TIME_UNIT = 5 -- render every X second
START=101 -- the begin id, to keep digits length the same, begin from 101 is suggested

-- the size to save images
b = bitmap 800 600 -- the bitmap buffer to save, without this memory will increase

-- set up rendering timeout to speed up rendering
--RendererClass.classes as string
renderers.current = RendererClass.classes[7]() -- 8
"current render: " +  (renderers.current as string)
vr = renderers.current
vr.max_render_time = 0.5 -- in minutes, 0.5 min ==30 seconds
--showproperties vr

-- setup vray to export depth maps
mgr = maxOps.GetCurRenderElementMgr() -- the cuurent render manager
mgr.removeallrenderelements() -- clean all previous rendering elements
mgr.addrenderelement (VRayZDepth elementname:(VRayZDepth as string)) -- add vraydepth element
zdepth=mgr.getRenderElement 0 -- get the vraydepth render element
zdepth.zdepth_min=0 -- set the nearest depth 
zdepth.zdepth_max=30000 -- set the farest depth
zdepth.enabled = true -- enable this rendering element
zdepth.filterOn = true -- ensure this vraydepth element filter turned on

for fid = START to (START+N-1) do
(
	-- render main camera with vray z-depth
	--render camera: $camMain renderMultiPassEffects:true renderElements:true to: b frame: ((fid-START)*TIME_UNIT)
	render camera: $camMain to: b frame: ((fid-START)*TIME_UNIT)
	b.filename = main_path +(fid as string) + "main.png"
	save b
	
	-- save depth map
	--max render last
	filename = depth_path + (fid as string) + "depth.png"
	"current buffer channel numer: " + (vrayVFBGetNumChannels() as string)
	vfbControl #setchannel 2
	vfbControl #saveimage filename
	
	-- render the aux camera
	render camera: $camAux to: b frame: ((fid-START)*TIME_UNIT)
	b.filename = aux_path + (fid as string) + "aux.png"
	save b
)
close b
"Job Done!"
-- end