import sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/harness_eval'))
from empty_memory import configure_empty_local_memory
class Tests(unittest.TestCase):
 def test_native_local_flags_and_zero_byte_index(self):
  with tempfile.TemporaryDirectory() as d:
   a=configure_empty_local_memory(['python','-m','rpent.cli.main','--model','openai-chat:gpt-6.1-sol'],d)
   self.assertEqual(a[-4:-2],['--memory-profile','local']);self.assertEqual(a[-2],'--memory-dir');p=Path(a[-1]);self.assertEqual((p/'MEMORY.md').stat().st_size,0);self.assertEqual(len([x for x in p.rglob('*') if x.is_file()]),1)
 def test_replaces_HF_flags_without_changing_model(self):
  with tempfile.TemporaryDirectory() as d:
   a=configure_empty_local_memory(['python','--memory-repo','old','--memory-profile=hf','--model','sol'],d)
   self.assertNotIn('--memory-repo',a);self.assertNotIn('--memory-profile=hf',a);self.assertEqual(a[a.index('--model')+1],'sol')
 def test_nonempty_memory_is_preserved_and_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'memory';p.mkdir();(p/'old.md').write_text('old experience')
   with self.assertRaises(ValueError):configure_empty_local_memory([],d)
   self.assertEqual((p/'old.md').read_text(),'old experience')
 def test_each_task_isolated(self):
  with tempfile.TemporaryDirectory() as d:
   a=configure_empty_local_memory([],Path(d)/'task0');b=configure_empty_local_memory([],Path(d)/'task1');self.assertNotEqual(a[-1],b[-1]);self.assertEqual(configure_empty_local_memory([],Path(d)/'task0'),a)
 def test_linked_memory_is_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   target=Path(d)/'target';target.mkdir();run=Path(d)/'task';run.mkdir();(run/'memory').symlink_to(target,target_is_directory=True)
   with self.assertRaises(ValueError):configure_empty_local_memory([],run)
if __name__=='__main__':unittest.main()
